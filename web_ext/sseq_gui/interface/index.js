import { MainDisplay, UnitDisplay } from './display.js';
import { ExtSseq } from './sseq.js';
import { dialog, renderLaTeX, download } from './utils.js';
import { openSocket } from './socket.js';

window.commandCounter = 0;
window.commandQueue = [];

/**
 * @description Processes a queue of commands and sends them to either main or unit
 * Sseq, depending on their type. It waits for all non-"Resolve" commands to be
 * processed before sending a "BlockRefresh" command to Sseq.
 */
function processCommandQueue() {
    if (window.commandQueue.length == 0) return;

    let commandText = '';
    const block = {
        recipients: ['Resolver', 'Sseq'],
        action: { BlockRefresh: { block: true } },
    };

    window.mainSseq.send(block);
    if (!window.mainSseq.isUnit) {
        window.unitSseq.send(block);
    }
    // If we are resolving, we should wait for it to finish resolving before we
    // can continue. For example, we don't want to add a differential when the
    // corresponding classes have not been generated.
    while (
        window.commandQueue.length > 0 &&
        !commandText.includes('"Resolve"')
    ) {
        commandText = window.commandQueue.pop();
        if (commandText.trim() == '') continue;

        try {
            const command = JSON.parse(commandText);
            if (command.sseq == 'Main') {
                window.mainSseq.send(command);
            } else {
                window.unitSseq.send(command);
            }
        } catch (e) {
            console.log('Unable to parse command ' + commandText);
            console.log(e);
            console.log(e.stack);
        }
    }
    block.action.BlockRefresh.block = false;

    window.mainSseq.send(block);
    if (!window.mainSseq.isUnit) {
        window.unitSseq.send(block);
    }
}

const url = new URL(document.location);
const params = {};
for (const [k, v] of url.searchParams.entries()) {
    params[k] = v;
}

if (params.module || params.module_json) {
    const maxDegree = parseInt(params.degree ? params.degree : 40);
    const algebra = params.algebra ? params.algebra : 'milnor';

    const action = params.module
        ? {
            Construct: {
                algebra_name: algebra,
                module_name: params.module,
            },
        }
        : {
            ConstructJson: {
                algebra_name: 'milnor',
                data: params.module_json,
            },
        };

    // Record this for the save functionality, since the wasm version modifies it
    window.constructCommand = {
        recipients: ['Resolver'],
        sseq: 'Main',
        action: action,
    };

    window.sendSocket = openSocket(
        [
            // The wasm version might mutate this object
            Object.assign({}, window.constructCommand),
            {
                recipients: ['Resolver'],
                sseq: 'Main',
                action: {
                    Resolve: {
                        max_degree: maxDegree,
                    },
                },
            },
        ],
        onMessage,
    );
} else {
    document.querySelector('#home').style.removeProperty('display');

    HTMLCollection.prototype.forEach = Array.prototype.forEach;
    const sections = document
        .querySelector('#home')
        .getElementsByTagName('section');

    sections.forEach(n => {
        // Traverses nested HTML elements and updates specific tags.

        n.children[1].children.forEach(a => {
            // Traverses and transforms HTML links.

            if (a.tagName == 'A') {
                a.innerHTML = renderLaTeX(a.innerHTML);
                a.href = `?module=${a.getAttribute('data')}&degree=40`;
            }
        });
    });
}

window.send = msg => {
    window.commandCounter += msg.recipients.length;
    if (window.display !== undefined)
        window.display.runningSign.style.removeProperty('display');

    window.sendSocket(msg);
};

/**
 * @description Processes incoming messages by parsing JSON data and invoking
 * corresponding handlers based on command types, either using `messageHandler` or
 * `window.sseq` properties. It also logs errors and relevant data if processing fails.
 * 
 * @param {Event} e - The event object triggered by a message receipt.
 */
function onMessage(e) {
    const data = JSON.parse(e.data);
    try {
        const command = Object.keys(data.action)[0];
        if (messageHandler[command]) {
            messageHandler[command](data.action[command], data);
        } else {
            switch (data.sseq) {
                case 'Main':
                    window.mainSseq['process' + command](data.action[command]);
                    break;
                case 'Unit':
                    window.unitSseq['process' + command](data.action[command]);
                    break;
                default:
            }
        }
    } catch (err) {
        console.log('Unable to process message');
        console.log(data);
        console.log(`Error: ${err}`);
        console.log(err.stack);
    }
}

/**
 * @description Constructs a command history by concatenating and stringifying an
 * array of objects, including the main sequence commands and additional unit sequence
 * commands if applicable, separated by newline characters.
 * 
 * @returns {string} A newline-separated concatenation of JSON-serialized arrays
 * containing objects with recipients, sseq, and action properties.
 */
function generateHistory() {
    const list = [window.constructCommand];
    list.push({
        recipients: ['Resolver'],
        sseq: 'Main',
        action: {
            Resolve: {
                max_degree: window.mainSseq.maxDegree,
            },
        },
    });
    if (!window.display.isUnit && window.unitSseq.maxDegree > 9) {
        list.push({
            recipients: ['Resolver'],
            sseq: 'Unit',
            action: {
                Resolve: {
                    max_degree: window.unitSseq.maxDegree,
                },
            },
        });
    }

    return list.concat(window.mainSseq.history).map(JSON.stringify).join('\n');
}

/**
 * @description Prompts the user to enter a file name, generates a text representation
 * of the history data using `generateHistory`, and then downloads it as a plain text
 * file with the specified name.
 */
function save() {
    dialog(
        'Save history',
        "<section class='input-row'><label>File name</label><input style='width: auto' required></input></section>",
        dialog =>
            download(
                dialog.querySelector('input').value,
                generateHistory(),
                'text/plain',
            ),
        'Save',
    );
}
window.save = save;

/**
 * @description Loads a history string into an array, removes empty lines and comments,
 * parses the first two commands, and assigns them to global variables. It also
 * populates a command queue with the remaining lines, which can be executed later.
 * 
 * @param {string} hist - History data loaded from file or socket connection.
 */
function loadHistory(hist) {
    const lines = hist.split('\n');
    // Do reverse loop because we are removing things from the array.
    for (let i = lines.length - 1; i >= 0; i--) {
        if (lines[i].startsWith('//') || lines[i].trim() === '') {
            lines.splice(i, 1);
        }
    }

    // First command is construct and second command is resolve
    window.constructCommand = JSON.parse(lines[0]);
    window.sendSocket = openSocket(
        lines.splice(0, 2).map(JSON.parse),
        onMessage,
    );

    lines.reverse();
    window.commandQueue = lines;
}

const messageHandler = {};
messageHandler.Resolving = (data, msg) => {
    if (msg.sseq == 'Unit') {
        if (!window.unitSseq) {
            window.unitSseq = new ExtSseq('Unit', 0);
        }
        window.unitSseq.processResolving(data);
        if (!window.unitDisplay) {
            window.unitDisplay = new UnitDisplay(
                'unitsseq-body',
                window.unitSseq,
            );
        }
        return;
    } else {
        if (!window.mainSseq) {
            window.mainSseq = new ExtSseq('Main', data.min_degree);
            window.mainSseq.isUnit = data.is_unit;
            if (data.is_unit) {
                window.unitSseq = window.mainSseq;
            }
        }
        window.mainSseq.processResolving(data);

        if (!window.display) {
            window.display = new MainDisplay(
                'main',
                window.mainSseq,
                data.is_unit,
            );
            window.display.runningSign.style.removeProperty('display');
        }
    }
};

messageHandler.Complete = () => {
    window.commandCounter--;
    if (window.commandCounter == 0) {
        window.display.runningSign.style.display = 'none';
        processCommandQueue();
    }
};

messageHandler.QueryBoundaryStringResult = m => {
    console.log(
        `Cocyle string for (t - s, s, idx) = (${m.t - m.s}, ${m.s}, ${m.idx}):`,
    );
    console.log(m.string);
};

messageHandler.QueryTableResult = m => {
    console.log(`Table for (t - s, s) = (${m.t - m.s}, ${m.s}):`);
    console.log(m.string);
};

messageHandler.Error = m => {
    console.error(m.message);
    dialog(
        'Fatal error encountered',
        `<section><pre>${m.message}</pre></section>`,
        () => { },
        'OK',
    );
};

// Set up upload button
document.getElementById('json-upload').addEventListener('change', () => {
    // Handles changes to a JSON upload field.

    const file = document.getElementById('json-upload').files[0];
    const fileReader = new FileReader();

    fileReader.onload = e => {
        try {
            // Remove whitespace to shorten URL
            const json = JSON.stringify(JSON.parse(e.target.result));
            window.location = `?module_json=${encodeURIComponent(json)}`;
        } catch (error) {
            alert('Invalid JSON file: ' + error.message);
            console.error('Error parsing JSON:', error);
        }
    };

    fileReader.readAsText(file, 'UTF-8');
});

document.getElementById('history-upload').addEventListener('change', () => {
    // Reads an uploaded file and hides a home page element.

    const file = document.getElementById('history-upload').files[0];

    const fileReader = new FileReader();
    fileReader.onload = e => {
        loadHistory(e.target.result);
    };

    fileReader.readAsText(file, 'UTF-8');
    document.querySelector('#home').style.display = 'none';
});
