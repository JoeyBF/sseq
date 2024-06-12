import { MainDisplay, UnitDisplay } from './display.js';
import { ExtSseq } from './sseq.js';
import { dialog, renderLaTeX, download } from './utils.js';
import { openSocket } from './socket.js';

window.commandCounter = 0;
window.commandQueue = [];

// Options modal
var modal = document.getElementById('optionsModal');

// Open the modal with the options button
/**
 * @description Sets the `display` CSS property of an HTML element with the ID "modal"
 * to "block".
 */
document.getElementById('optionsButton').addEventListener('click', () => {
    modal.style.display = 'block';
});

// When the user clicks anywhere outside of the modal, close it
window.onclick = event => {
    if (event.target == modal) {
        modal.style.display = 'none';
    }
};

/**
 * @description Enables or disables a text input field based on the selected state
 * of a checkbox.
 */
document
    .getElementById('saveDirectoryCheckbox')
    .addEventListener('change', function () {
        var directoryInput = document.getElementById('saveName');
        if (this.checked) {
            directoryInput.disabled = false;
        } else {
            directoryInput.disabled = true;
        }
    });

// Populate the datalist with existing saves
/**
 * @description Clears any existing options in a `<datalist>` element with ID
 * `'saveNames'`, then adds a new option for each `save` parameter provided, with
 * each option's value set to the corresponding `save`.
 * 
 * @param { array } saves - an array of saves for which options will be generated and
 * added to the `datalist`.
 */
function populateSaveNames(saves) {
    const datalist = document.getElementById('saveNames');
    datalist.innerHTML = ''; // Clear existing options
    saves.forEach(save => {
        const option = document.createElement('option');
        option.value = save;
        datalist.appendChild(option);
    });
}

// Call this function with the list of saves when the page loads or the saves are updated
populateSaveNames([]);

// Use this function when you need the save name
console.log(getSaveName());

/**
 * @description Manages the queue of JSON commands and sends them to the corresponding
 * sequence servers (main or unit) based on their type and status of resolution.
 * 
 * @returns { undefined` value since no return statement has been implemented inside
 * the function } a series of SSEQ commands sent to the Main and Unit queues based
 * on the contents of the command queue.
 * 
 * 		- `block`: An object with `recipients` and `action`. The `recipients` array
 * contains the IDs of the sequencing units to which the block should be sent, while
 * the `action` object contains a single property, `BlockRefresh`, which is an object
 * with two properties: `block` (a boolean indicating whether the block should be
 * displayed) and `reason` (a string explaining why the block was refreshed).
 * 		- `commandText`: A string representing the text of the command that was popped
 * from the command queue.
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
    const saveName = params.save ? params.save : null;
    const algebra = params.algebra ? params.algebra : 'milnor';

    const action = params.module
        ? {
            Construct: {
                algebra_name: algebra,
                module_name: params.module,
                save_name: saveName,
            },
        }
        : {
            ConstructJson: {
                algebra_name: 'milnor',
                data: params.module_json,
                save_name: saveName,
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
        n.children[1].children.forEach(a => {
            if (a.tagName == 'A') {
                a.innerHTML = renderLaTeX(a.innerHTML);
                a.addEventListener('click', function (event) {
                    event.preventDefault();
                    window.location.href = `?module=${a.getAttribute(
                        'data',
                    )}&degree=40&save=${document.getElementById('saveName').value
                        }`;
                });
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
 * @description Parses a JSON message, identifies the command from the message's data
 * key, and executes the corresponding handler function with the data provided.
 * 
 * @param { object } e - message object passed through the event listener and provides
 * the necessary data for processing.
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
 * @description Populates a list with historical commands and actions, then joins
 * them into a single string of JSON objects using the `JSON.stringify()` method.
 * 
 * @returns { array } a concatenation of JSON-stringified historical data for various
 * sequences, including `window.mainSseq` and `window.unitSseq`.
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
 * @description Opens a dialog box displaying a input field for the file name, followed
 * by an action to download a generated history file as a text plain file.
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
 * @description Loads history data from a string and separates it into commands and
 * resolve messages. It also sets the `window.constructCommand` and `window.sendSocket`.
 * 
 * @param { string } hist - history of commands and their responses, which is split
 * into an array of strings and then processed in the function.
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
/**
 * @description Manages sequence data passed from parent functions, determining if
 * it belongs to a unit or main sequence based on `msg.sseq`. If a unit sequence, it
 * initializes a `UnitDisplay` instance, else an `MainDisplay` instance. Processes
 * resolving and updates display as needed.
 * 
 * @param { object } data - resolving event that triggered the function, providing
 * information such as the sequence number and minimum degree of the event.
 * 
 * @param { object } msg - message passed from the event listener, which determines
 * the branch of code executed based on its `sseq` property value.
 * 
 * @returns { undefined` value } a process of resolving sequences based on the input
 * data.
 * 
 * 		- `window.unitSseq`: This is an instance of `ExtSseq` with a `name` of `'Unit'`
 * and an `id` of `'Unitsseq-body'`. It represents a sequence of units for the current
 * degree.
 * 		- `window.mainSseq`: This is an instance of `ExtSseq` with a `name` of `'Main'`
 * and an `id` of `'Main-body'`. It represents the main sequence of symbols for the
 * current degree.
 * 		- `window.unitDisplay`: This is an instance of `UnitDisplay` with an `id` of
 * `'Unitsseq-body'`. It is responsible for rendering the units sequence.
 * 		- `window.mainDisplay`: This is an instance of `MainDisplay` with an `id` of
 * `'Main'`. It is responsible for rendering the main sequence of symbols.
 * 		- `data.min_degree`: This is the minimum degree of the symbol being processed.
 * 		- `data.is_unit`: This is a boolean indicating whether the symbol is a unit. If
 * it is, then `window.unitSseq` will be set to `window.mainSseq`.
 */
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

/**
 * @description Decrements `window.commandCounter` and checks if it is equal to 0.
 * If it is, it hides the `window.display.runningSign` element and executes the
 * `processCommandQueue()` function.
 */
messageHandler.Complete = () => {
    window.commandCounter--;
    if (window.commandCounter == 0) {
        window.display.runningSign.style.display = 'none';
        processCommandQueue();
    }
};

messageHandler.QueryCocycleStringResult = m => {
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
/**
 * @description Reads the contents of a selected JSON file using `FileReader` and
 * replaces them with a URL parameter for further processing.
 */
document.getElementById('json-upload').addEventListener('change', () => {
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

/**
 * @description Loads a file selected by a user from an HTML input element with ID
 * "history-upload" into the history of a web application using `FileReader`.
 */
document.getElementById('history-upload').addEventListener('change', () => {
    const file = document.getElementById('history-upload').files[0];

    const fileReader = new FileReader();
    fileReader.onload = e => {
        loadHistory(e.target.result);
    };

    fileReader.readAsText(file, 'UTF-8');
    document.querySelector('#home').style.display = 'none';
});
