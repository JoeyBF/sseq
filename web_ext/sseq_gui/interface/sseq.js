import { dialog } from './utils.js';
import { svgNS } from './chart.js';
import './components.js';

export const MIN_PAGE = 2;
const CHART_STYLE = `
.class-error {
    fill: #a6001a;
}
.class-done {
    fill: gray;
}
.class-group:hover > circle {
    fill: red;
}
.highlight > circle, .selected > circle {
    fill: red;
}
.differential {
    stroke-width: 0.02;
    stroke: orange;
}
.structline {
    stroke-width: 0.02;
    stroke: black;
}
.d2 {
    stroke: cyan;
}
.d3 {
    stroke: red;
}
.d4 {
    stroke: green;
}
.d5 {
    stroke: blue;
}
`;

const KEEP_LOG = new Set([
    'AddDifferential',
    'AddProductType',
    'AddProductDifferential',
    'AddPermanentClass',
    'SetClassName',
]);

/**
 * @description Represents a two-dimensional vector space with adjustable minimum
 * degree. It provides methods for setting and retrieving values at specific indices
 * within this space, while ensuring that the underlying data structure is initialized
 * to accommodate all possible indices.
 */
export class BiVec {
    /**
     * @description Initializes an instance with two parameters: `minDegree` and `data`.
     * If `data` is provided, it assigns the value to the `data` property; otherwise, it
     * defaults to an empty array. The `minDegree` parameter sets the minimum degree of
     * the BiVec object.
     * 
     * @param {number} minDegree - Set as the minimum degree for an object.
     * 
     * @param {(string | object)[]} data - Optional. It initializes an array for storing
     * data.
     */
    constructor(minDegree, data) {
        this.data = data ? data : [];
        this.minDegree = minDegree;
    }
    /**
     * @description Sets a value at a specific location (x, y) in the internal data
     * structure (`this.data`). If necessary, it expands the array to accommodate new
     * elements by pushing empty arrays until its length is greater than or equal to x -
     * minDegree.
     * 
     * @param {number} x - 0-indexed row index for data storage.
     * 
     * @param {number} y - 0-indexed.
     * 
     * @param {any} data - Used to store a value at a specific location.
     */
    set(x, y, data) {
        while (this.data.length <= x - this.minDegree) {
            this.data.push([]);
        }
        this.data[x - this.minDegree][y] = data;
    }
    /**
     * @description Retrieves a value from a nested data structure based on provided
     * coordinates (x, y). The x-coordinate is adjusted by subtracting the `minDegree`
     * property to access the corresponding nested object or array.
     * 
     * @param {number} x - 0-based index for accessing data structure.
     * 
     * @param {number} y - An index for accessing data within the returned value.
     * 
     * @returns {any} Retrieved from a multi-dimensional array-like data structure at the
     * specified indices `x` and `y`.
     */
    get(x, y) {
        return this.data?.[x - this.minDegree]?.[y];
    }
}

/**
 * @description Represents an extended spectral sequence, providing a graphical
 * interface for manipulating and visualizing its structure. It enables users to add
 * classes, differentials, and products, as well as resolve further, query boundaries,
 * and show/hide structural lines.
 */
export class ExtSseq {
    /**
     * @description Initializes an object with various properties and sets up event
     * listeners. It creates several data structures, including `BiVec` instances for
     * classes, decompositions, and products, as well as a DOM element for rendering charts.
     * 
     * @param {string} name - Used to set the name of an object.
     * 
     * @param {number} minDegree - Used to set the minimum degree for various data structures.
     */
    constructor(name, minDegree) {
        this.minDegree = minDegree;
        this.maxDegree = minDegree;

        this.history = [];
        this.redoStack = [];
        this.name = name;

        this.vanishingSlope = '1/2';
        this.vanishingIntercept = 1;
        this.visibleStructlines = new Set(['h_0', 'a_0', 'h_1', 'h_2']);

        this.classes = new BiVec(minDegree);
        this.classState = new BiVec(minDegree);
        this.permanentClasses = new BiVec(minDegree);
        this.classNames = new BiVec(minDegree);
        this.decompositions = new BiVec(minDegree);
        this.products = new Map();
        this.trueDifferentials = new BiVec(minDegree);

        this.chart = document.createElement('paged-chart');
        this.chart.addStyle(CHART_STYLE);
        this.chart.setAttribute('minx', minDegree);
        this.chart.addEventListener('click', () => this.select(null));

        this.selected = null;
        this.refreshPanel = undefined;
        this.onClick = undefined;

        this.chart.newPage();

        this._onClassClick = this.__onClassClick.bind(this);
    }

    /**
     * @description Returns the current page number plus a constant `MIN_PAGE`, providing
     * an offset from the base page number to obtain the actual page number used by the
     * chart.
     * 
     * @returns {number} The sum of `chart.page` and a constant `MIN_PAGE`.
     */
    get page() {
        return this.chart.page + MIN_PAGE;
    }

    /**
     * @description Creates a new page by cloning the previous page and appending it to
     * the chart. It also sets an event listener for click events on nodes with the class
     * 'class-group' within the cloned page.
     */
    newPage() {
        const prevPage = this.chart.pages[this.chart.pages.length - 1];
        const page = prevPage.cloneNode(true);
        for (const node of page.getElementsByClassName('class-group')) {
            node.childNodes.forEach(x => (x.onclick = this._onClassClick));
        }
        this.chart.appendPage(page);
    }

    /**
     * @description Sends data to an unknown destination, possibly logging it depending
     * on the presence of certain action keys and the value of the `log` parameter. The
     * data is modified by adding a sequence number before being sent.
     * 
     * @param {object} data - Used to send data to somewhere.
     * 
     * @param {boolean} log - Used to log the data.
     */
    send(data, log = true) {
        if (KEEP_LOG.has(Object.keys(data.action)[0])) {
            if (log) {
                this.history.push(data);
            }
        }

        data.sseq = this.name;
        window.send(data);
    }

    /**
     * @description Removes an action from its history, prompting the user to confirm
     * before doing so. If confirmed, it sends a "Clear" action and then resends all
     * previous actions to restore the state.
     * 
     * @param {object} msg - Used to store an action message from the history stack.
     */
    removeHistoryItem(msg) {
        const actionKey = Object.keys(msg['action'])[0];
        dialog(
            'Undo action',
            `<section>Undoing action:
            <pre>${actionKey}: ${JSON.stringify(
                msg['action'][actionKey],
                null,
                4,
            )}</pre>
            </section>`,
            () => {
                // "Undo" an action.

                msg = JSON.stringify(msg);
                this.history = this.history.filter(
                    m => JSON.stringify(m) != msg,
                );

                this.block();
                this.send({
                    recipients: ['Sseq'],
                    action: { Clear: {} },
                });

                for (const msg of this.history) {
                    this.send(msg, false);
                }
                this.refreshPanel?.();
                this.block(false);
            },
            'Undo',
        );
    }

    /**
     * @description Sends a request to refresh the block with the specified value for the
     * parameter `block`, targeting recipients 'Sseq'.
     * 
     * @param {boolean} block - Used to determine whether to block or unblock a user.
     */
    block(block = true) {
        this.send({
            recipients: ['Sseq'],
            action: { BlockRefresh: { block: block } },
        });
    }

    /**
     * @description Undoes the previous action by removing it from the history and redoing
     * it on top of the redo stack, then clears all messages, refreshes the panel, and
     * releases any blocks.
     */
    undo() {
        this.redoStack.push(this.history.pop());

        this.block();
        this.send({
            recipients: ['Sseq'],
            action: { Clear: {} },
        });

        for (const msg of this.history) {
            this.send(msg, false);
        }
        this.refreshPanel?.();
        this.block(false);
    }

    /**
     * @description Pops and sends an item from the `redoStack`, effectively redoing the
     * last undone action.
     */
    redo() {
        this.send(this.redoStack.pop());
    }

    /**
     * @description Sends a message to a specified recipient (`'Sseq'`) with an action
     * to add a permanent CSS class to an element. The method takes three parameters:
     * `x`, `y`, and `target`, which represent the coordinates and the class to be added,
     * respectively.
     * 
     * @param {number} x - The x-coordinate of the target element.
     * 
     * @param {number} y - Used to specify a y-coordinate.
     * 
     * @param {string} target - Used to specify the target class name.
     */
    addPermanentClass(x, y, target) {
        this.send({
            recipients: ['Sseq'],
            action: {
                AddPermanentClass: {
                    x: x,
                    y: y,
                    class: target,
                },
            },
        });
    }

    /**
     * @description Converts a page basis to an E2 basis, given coordinates (x, y), radius
     * r, and coefficients c. It multiplies corresponding elements, takes modulo p, and
     * returns the resulting vector.
     * 
     * @param {number} r - Used for calculations.
     * 
     * @param {number} x - Used as an index for accessing class data.
     * 
     * @param {number} y - Used to access specific elements in an array.
     * 
     * @param {number[]} c - Used for coefficient calculation.
     * 
     * @returns {number[]} A list of coefficients of a polynomial over the finite field
     * with modulus `p`, representing the result of transforming basis vectors from
     * page-basis to e2-basis.
     */
    pageBasisToE2Basis(r, x, y, c) {
        const len = this.classes.get(x, y)[0].length;
        const pageBasis = this.getClasses(x, y, r);

        const result = [];
        for (let i = 0; i < len; i++) {
            result.push(0);
        }
        for (let i = 0; i < pageBasis.length; i++) {
            const coef = c[i];
            for (let j = 0; j < len; j++) {
                result[j] += coef * pageBasis[i][j];
            }
        }
        for (let i = 0; i < len; i++) {
            result[i] = result[i] % this.p;
        }
        return result;
    }

    /**
     * @description Creates an interactive dialog box for users to input differential
     * equations between two pages of an extension sequence. The input fields are defined
     * based on the dimensions of the source and target vectors, which depend on the
     * current page.
     * 
     * @param {[number, number]} source - 2D coordinates of a point.
     * 
     * @param {[number, number]} target - Used as the target point for calculating differential.
     */
    addDifferentialInteractive(source, target) {
        const page = target[1] - source[1];
        const sourceDim = this.getClasses(source[0], source[1], page).length;
        const targetDim = this.getClasses(target[0], target[1], page).length;

        dialog(
            `Input differential at (${source[0]}, ${source[1]})`,
            `<section style="text-align: center">
                ${katex.renderToString(`d_{${page}}`)}
                <input name="source" is="class-input"
                    title="Express source in E${page} page basis"
                    length="${sourceDim}" p=${this.p}></input>
                =
                <input name="target" is="class-input"
                    title="Express target in E${page} page basis"
                    length="${targetDim}" p=${this.p}></input>
            </section>`,
            dialog => {
                // Opens a dialog.

                const sourceVec = this.pageBasisToE2Basis(
                    page,
                    source[0],
                    source[1],
                    eval(dialog.querySelector("input[name='source']").value),
                );
                const targetVec = this.pageBasisToE2Basis(
                    page,
                    source[0] - 1,
                    source[1] + page,
                    eval(dialog.querySelector("input[name='target']").value),
                );

                this.addDifferential(
                    page,
                    source[0],
                    source[1],
                    sourceVec,
                    targetVec,
                );
            },
        );
    }

    /**
     * @description Sends a request to set a new class name for an object at specific
     * coordinates (x, y) and index idx. The new name is provided as a parameter. The
     * recipients of this message are specified by 'Sseq'.
     * 
     * @param {number} x - Used to set the class name.
     * 
     * @param {number} y - Unspecified in this description.
     * 
     * @param {number} idx - Used to set class name at specific index.
     * 
     * @param {string} name - Used to set the name of a class.
     */
    setClassName(x, y, idx, name) {
        this.send({
            recipients: ['Sseq'],
            action: { SetClassName: { x: x, y: y, idx: idx, name: name } },
        });
    }

    // addProductInteractive takes in the number of classes in bidegree (x, y), because this should be the number of classes in the *unit* spectral sequence, not the main spectral sequence
    /**
     * @description Opens a dialog to add a product at a specified position (x, y). The
     * dialog contains input fields for product name and E2 page basis class, and an
     * option for permanent product. On submission, it sends an action with the provided
     * information to recipients 'Sseq' and 'Resolver'.
     * 
     * @param {number} x - Used as an x-coordinate.
     * 
     * @param {number} y - Used to specify the y-coordinate of the product's location.
     * 
     * @param {number} num - Used to set the length of an input field.
     */
    addProductInteractive(x, y, num) {
        dialog(
            `Add product at (${x}, ${y})`,
            `<section style="display: flex; justify-content: center; align-items: center; gap: 1em">
                <katex-input style='text-align: right' width='5em' input placeholder='name' title='Name of product'></katex-input>
                =
                <input name='class' is='class-input' title='Class in E2 page basis' length='${num}' p=${this.p}></input>
            </section>
            <section style="display: flex; justify-content: center; align-items: center; gap: 1em">
                Permanent <checkbox-switch checked></checkbox-switch>
            </section>`,
            dialog => {
                // Opens a dialogue to add product at specific coordinates and send relevant data.

                this.send({
                    recipients: ['Sseq', 'Resolver'],
                    action: {
                        AddProductType: {
                            permanent:
                                dialog.querySelector('checkbox-switch')
                                    .checked === true,
                            x: x,
                            y: y,
                            class: eval(
                                dialog.querySelector("input[name='class']")
                                    .value,
                            ),
                            name: dialog.querySelector('katex-input').value,
                        },
                    },
                });
            },
        );
    }

    /**
     * @description Creates an interactive dialog to add product differential at a specified
     * position on a grid, allowing users to input source and target expressions and
     * names, then sends this data to the main Sseq module for processing.
     * 
     * @param {number} sourceX - Used to specify the x-coordinate of the source product.
     * 
     * @param {number} sourceY - Utilized as an input to calculate the target class.
     * 
     * @param {number} page - Used to calculate coordinates for adding products differentially.
     * 
     * @param {string[][]} sourceClass - Used to define the source class.
     * 
     * @param {(number | string[])[]} targetClass - Used to express target in E page basis.
     */
    addProductDifferentialInteractive(
        sourceX,
        sourceY,
        page,
        sourceClass,
        targetClass,
    ) {
        const sourceDim = this.getClasses(sourceX, sourceY, MIN_PAGE).length;
        const targetDim = this.getClasses(
            sourceX - 1,
            sourceY + page,
            MIN_PAGE,
        ).length;
        dialog(
            `Add product differential at (${sourceX}, ${sourceY})`,
            `<section style="text-align: center">
                ${katex.renderToString(`d_{${page}}`)}
                <input name="source" is="class-input"
                    title="Express source in E${page} page basis"
                    length="${sourceDim}" p=${this.p}
                    value="${sourceClass ? '[' + sourceClass.join(', ') + ']' : ''
            }"
                ></input>
                =
                <input name="target" is="class-input"
                    title="Express target in E${page} page basis"
                    length="${targetDim}" p=${this.p}
                    value="${targetClass ? '[' + targetClass.join(', ') + ']' : ''
            }"
                ></input>
            </section>
            <section>
                <div class="input-row">
                    <label style="width: 6em">Source name</label>
                    <katex-input width="10em" input title='Name of source' placeholder='source name' name='source-name'></katex-input>
                </div>
                <div class="input-row">
                    <label style="width: 6em">Target name</label>
                    <katex-input width="10em" input title='Name of target' placeholder='target name' name='target-name'></katex-input>
                </div>
            </section>`,
            dialog => {
                // Adds product differential.

                window.mainSseq.send({
                    recipients: ['Sseq', 'Resolver'],
                    action: {
                        AddProductDifferential: {
                            source: {
                                permanent: false,
                                x: sourceX,
                                y: sourceY,
                                class: eval(
                                    dialog.querySelector("input[name='source']")
                                        .value,
                                ),
                                name: dialog
                                    .querySelector(
                                        "katex-input[name='source-name']",
                                    )
                                    .value.trim(),
                            },
                            target: {
                                permanent: false,
                                x: sourceX - 1,
                                y: sourceY + page,
                                class: eval(
                                    dialog.querySelector("input[name='target']")
                                        .value,
                                ),
                                name: dialog
                                    .querySelector(
                                        "katex-input[name='target-name']",
                                    )
                                    .value.trim(),
                            },
                        },
                    },
                });
            },
        );
    }

    /**
     * @description Allows adding a permanent class at a specified position (x, y) on a
     * grid-like structure. The method offers interactive options for selecting a class
     * to add, based on existing classes in that position.
     * 
     * @param {number} x - 1D spatial coordinate.
     * 
     * @param {number} y - 2D coordinate's y-axis value.
     */
    addPermanentClassInteractive(x, y) {
        const classes = this.classes.get(x, y);

        const last = classes[classes.length - 1];
        if (last.length == 0) {
            dialog(
                `Add permanent class at (${x}, ${y})`,
                '<section>There are no surviving classes</section>',
                () => { },
                'OK',
            );
        } else if (classes[0].length == 1) {
            this.addPermanentClass(x, y, classes[0][0]);
        } else {
            dialog(
                `Add permanent class at (${x}, ${y})`,
                `<section class="input-row">
                    <label>Class</label><input is="class-input" p="${this.p}" length="${classes[0].length}"></input>
                </section>`,
                dialog =>
                    this.addPermanentClass(
                        x,
                        y,
                        eval(dialog.querySelector('input').value),
                    ),
            );
        }
    }

    /**
     * @description Sends a request to add a differential link between two nodes in a
     * network, specifying the source node coordinates (`source_x`, `source_y`), source
     * and target node identifiers (`source`, `target`), and other properties.
     * 
     * @param {number} r - Used for specifying a value.
     * 
     * @param {number} source_x - Part of the differential coordinates.
     * 
     * @param {number} source_y - Used to represent y-coordinate of the source.
     * 
     * @param {object} source - Used to represent the source node or edge.
     * 
     * @param {string} target - Intended to specify the target point for the differential.
     */
    addDifferential(r, source_x, source_y, source, target) {
        this.send({
            recipients: ['Sseq'],
            action: {
                AddDifferential: {
                    r: r,
                    x: source_x,
                    y: source_y,
                    source: source,
                    target: target,
                },
            },
        });
    }

    /**
     * @description Resolves further to obtain a new maximum degree from the user if the
     * input is not an integer, and then sends the updated maximum degree to the 'Resolver'.
     * 
     * @param {number | string} newmax - Used to update the maximum degree.
     */
    resolveFurther(newmax) {
        if (Number.isInteger(newmax)) {
            this.maxDegree = newmax;
            this.send({
                recipients: ['Resolver'],
                action: {
                    Resolve: {
                        max_degree: newmax,
                    },
                },
            });
            return;
        }

        dialog(
            'Resolve further',
            `<section style="input-row">
                <label>New maximum degree</label>
                <input style="width: 5em" type="number" value="${this.maxDegree + 10
            }">
            </section>`,
            dialog => {
                // Resolves further.

                newmax = parseInt(dialog.querySelector('input').value);
                if (newmax <= this.maxDegree) {
                    return;
                }
                this.maxDegree = newmax;

                this.send({
                    recipients: ['Resolver'],
                    action: {
                        Resolve: {
                            max_degree: newmax,
                        },
                    },
                });
            },
            'Resolve',
        );
    }

    /**
     * @description Retrieves a set of classes from a 2D array at a specified position
     * (x, y). If the set exists, it sends a query to the 'Resolver' for each element in
     * the set, providing the row number (y), column number plus row number (x+y), and index.
     * 
     * @param {number} x - Used to identify a specific location.
     * 
     * @param {string} y - Used as an input for boundary string querying.
     */
    queryBoundaryString(x, y) {
        const classes = this.classes.get(x, y);
        if (!classes) return;

        const len = classes[0].length;

        for (let i = 0; i < len; i++) {
            this.send(
                {
                    recipients: ['Resolver'],
                    action: {
                        QueryBoundaryString: {
                            s: y,
                            t: x + y,
                            idx: i,
                        },
                    },
                },
                false,
            );
        }
    }

    /**
     * @description Sends a query to 'Resolver' when the parameter `y` is non-negative,
     * requesting data for a table with specified start row (`s`) and total rows (`t`).
     * If `y` is negative, no action is taken.
     * 
     * @param {number} x - Used as an input value for table querying.
     * 
     * @param {number} y - Used to determine query logic.
     */
    queryTable(x, y) {
        if (y < 0) {
            return;
        }

        this.send(
            {
                recipients: ['Resolver'],
                action: {
                    QueryTable: {
                        s: y,
                        t: x + y,
                    },
                },
            },
            false,
        );
    }

    /**
     * @description Initializes instance variables `p` and `maxDegree` from input data,
     * then calls the `updateDegrees` method to update degrees based on the initialized
     * values.
     * 
     * @param {object} data - Used to initialize class properties.
     */
    processResolving(data) {
        this.p = data.p;
        this.maxDegree = data.max_degree;
        this.updateDegrees();
    }

    /**
     * @description Updates the attributes of an SVG chart, setting its minimum and maximum
     * x-coordinates to `minDegree` and `maxDegree`, respectively. It also calculates and
     * sets the maximum y-coordinate based on a mathematical formula involving `vanishingSlope`
     * and `vanishingIntercept`.
     */
    updateDegrees() {
        this.chart.setAttribute('minx', this.minDegree);
        this.chart.setAttribute('maxx', this.maxDegree);

        this.chart.setAttribute(
            'maxy',
            Math.ceil(
                (this.maxDegree - this.minDegree) * eval(this.vanishingSlope) +
                1 +
                eval(this.vanishingIntercept),
            ),
        ); // We trust our inputs *so* much.
    }

    /**
     * @description Removes elements from a specified part of an HTML document based on
     * their class name, which follows a specific pattern. The method iterates through
     * all such elements and clears them.
     * 
     * @param {string} type - Used to filter elements by class name.
     * 
     * @param {string} x - Used to identify specific elements.
     * 
     * @param {number} y - Used to specify a unique identifier for an element.
     * 
     * @param {object} p - Used to define the target element for class name selection.
     */
    clearOld(type, x, y, p) {
        if (p === undefined) p = this.chart.contents;

        const classes = Array.from(
            p.getElementsByClassName(`${type}-${x}-${y}`),
        );
        for (const c of classes) {
            c.remove();
        }
    }

    /**
     * @description Calculates and returns the position on an axis given the current
     * position `x`, dimensionality `dim`, and index `i`. It applies an offset to `x`
     * based on `i` and the dimensionality, effectively shifting the position along the
     * axis.
     * 
     * @param {number} x - Likely an initial position.
     * 
     * @param {number} dim - 2D dimension.
     * 
     * @param {number} i - 1-indexed.
     * 
     * @returns {number} A position calculated from the input parameters.
     */
    static getPosition(x, dim, i) {
        const offset = i - (dim - 1) / 2;
        return x + offset * 0.3;
    }

    /**
     * @description Intercepts click events on elements with data-x and data-y attributes,
     * extracts these coordinates, and updates the state by calling the `select` method
     * with the extracted values.
     * 
     * @param {Event} e - Used to capture the click event.
     */
    __onClassClick(e) {
        e.stopPropagation();
        const x = parseInt(e.target.parentNode.getAttribute('data-x'));
        const y = parseInt(e.target.parentNode.getAttribute('data-y'));
        this.select([x, y]);
    }

    /**
     * @description Updates the visualization of classes for a given data set by clearing
     * and recreating graphical representations based on new data, taking into account
     * the state of classes (done or error) and applying visual styling accordingly.
     * 
     * @param {object} data - Expected to contain page data.
     */
    processSetClass(data) {
        const x = data.x;
        const y = data.y;

        const oldClasses = this.classes.get(x, y);
        // classes is a list, and each member of the list corresponds to a
        // page. Each page itself is a list of classes.
        this.classes.set(x, y, data.classes);
        this.classState.set(x, y, data.state);
        this.permanentClasses.set(x, y, data.permanents);
        this.classNames.set(x, y, data.class_names);
        this.decompositions.set(x, y, data.decompositions);

        for (const [r, page] of this.chart.pages.entries()) {
            const num = this.getClasses(x, y, r + MIN_PAGE).length;
            const oldNum =
                ExtSseq.getPage(oldClasses, r + MIN_PAGE)?.length || 0;

            let classname = 'class';
            if (data.state === 'Done') {
                classname = 'class-done';
            } else if (data.state === 'Error') {
                classname = 'class-error';
            }

            if (oldNum === num) {
                if (num > 0) {
                    const grp = page.getElementsByClassName(
                        `class-group-${x}-${y}`,
                    )[0];

                    for (const child of grp.children) {
                        child.setAttribute('class', classname);
                    }
                }
                continue;
            }

            this.clearOld('class-group', x, y, page);
            if (num == 0) {
                continue;
            }
            const grp = document.createElementNS(svgNS, 'g');
            grp.classList.add(`class-group`);
            grp.classList.add(`class-group-${x}-${y}`);
            grp.setAttribute('data-x', x);
            grp.setAttribute('data-y', y);
            for (let i = 0; i < num; i++) {
                const node = document.createElementNS(svgNS, 'circle');
                node.setAttribute('cx', ExtSseq.getPosition(x, num, i));
                node.setAttribute('cy', -y);
                node.setAttribute('r', 0.1);
                node.setAttribute('class', classname);

                const title = document.createElementNS(svgNS, 'title');
                title.textContent = `(${x}, ${y})`;
                node.appendChild(title);

                node.onclick = this._onClassClick;
                grp.appendChild(node);
            }
            page.appendChild(grp);
        }
        if (this.hasSelected(x, y)) {
            this.select([x, y]);
        }
    }

    /**
     * @description Checks if a cell at coordinates `(x, y)` is selected by comparing it
     * with the current `selected` property, which is an array of two values representing
     * the currently selected cell's coordinates.
     * 
     * @param {number} x - A coordinate value.
     * 
     * @param {number} y - Part of coordinates.
     * 
     * @returns {boolean} True if a selection exists and its coordinates match the specified
     * x and y values, false otherwise.
     */
    hasSelected(x, y) {
        return (
            this.selected !== null &&
            this.selected[0] == x &&
            this.selected[1] == y
        );
    }

    /**
     * @description Updates the selection state by removing the previous selection and
     * adding a new one. It also calls the `onClick` and `refreshPanel` methods if they
     * are defined.
     * 
     * @param {(number|string)[]} select - Selected element or elements to be highlighted
     * on the chart.
     */
    select(select) {
        this.chart.shadowRoot
            .querySelectorAll(`.selected`)
            .forEach(x => x.classList.remove('selected'));
        const oldSelect = this.selected;
        this.selected = select;
        if (select !== null) {
            this.chart.shadowRoot
                .querySelectorAll(`.class-group-${select[0]}-${select[1]}`)
                .forEach(x => x.classList.add('selected'));
        }
        this.onClick?.(oldSelect);
        this.refreshPanel?.();
    }

    /**
     * @description Draws a matrix of lines or curves on an SVG canvas, based on the input
     * matrix and bend parameter. It generates elements for the SVG document and yields
     * them for further processing. The bend parameter determines whether to draw straight
     * lines or curved paths.
     * 
     * @param {object[][]} matrix - 2D matrix of values to be drawn on an SVG.
     * 
     * @param {number} sourceX - Used to calculate positions of lines or curves.
     * 
     * @param {number} targetX - Used to determine the target x-coordinate for drawing lines.
     * 
     * @param {number} sourceY - Used to specify the y-coordinate of the source point.
     * 
     * @param {number} targetY - Utilized as the y-coordinate for the target position.
     * 
     * @param {number} bend - Used to control curve bending during drawing process.
     */
    static *drawMatrix(matrix, sourceX, targetX, sourceY, targetY, bend = 0) {
        for (const [sourceIdx, row] of matrix.entries()) {
            for (const [targetIdx, val] of row.entries()) {
                if (val === 0) {
                    continue;
                }
                const x1 = ExtSseq.getPosition(
                    sourceX,
                    matrix.length,
                    sourceIdx,
                );
                const x2 = ExtSseq.getPosition(targetX, row.length, targetIdx);
                if (bend === 0) {
                    const line = document.createElementNS(svgNS, 'line');
                    line.setAttribute('x1', x1);
                    line.setAttribute('x2', x2);
                    line.setAttribute('y1', -sourceY);
                    line.setAttribute('y2', -targetY);
                    yield line;
                } else {
                    const midX = (x1 + x2) / 2;
                    const midY = (sourceY + targetY) / 2;
                    const controlX = midX - ((targetY - sourceY) * bend) / 100;
                    const controlY = midY + ((x2 - x1) * bend) / 100;
                    const path = document.createElementNS(svgNS, 'path');

                    path.style.fill = 'none';
                    path.setAttribute(
                        'd',
                        `M ${x1} ${-sourceY} Q ${controlX} ${-controlY}, ${x2} ${-targetY}`,
                    );
                    yield path;
                }
            }
        }
    }

    /**
     * @description Processes differential data by updating the chart pages, clearing old
     * data, drawing new matrices, and adding them to the pages. It also refreshes a panel
     * if a specific selection is made.
     * 
     * @param {object} data - Used to set differentials.
     */
    processSetDifferential(data) {
        const x = data.x;
        const y = data.y;

        while (this.chart.pages.length <= data.differentials.length) {
            this.newPage();
        }
        this.trueDifferentials.set(x, y, data.true_differentials);

        this.clearOld('differential', x, y);

        for (const [r, diffs] of data.differentials.entries()) {
            const page = this.chart.pages[r];
            for (const diff of ExtSseq.drawMatrix(
                diffs,
                x,
                x - 1,
                y,
                y + r + MIN_PAGE,
            )) {
                diff.classList.add(`differential`);
                diff.classList.add(`differential-${x}-${y}`);
                diff.classList.add(`d${r + MIN_PAGE}`);
                // Go under classes
                page.insertBefore(diff, page.firstChild);
            }
        }
        if (this.hasSelected(x, y)) {
            this.refreshPanel?.();
        }
    }

    /**
     * @description Iterates through an array of structlines, updating product information
     * and matrices, and then redraws the corresponding lines based on the updated matrices.
     * It also updates the visibility of these lines if necessary.
     * 
     * @param {object} data - Used to process struct lines.
     */
    processSetStructline(data) {
        const x = data.x;
        const y = data.y;

        for (const mult of data.structlines) {
            if (!this.products.has(mult.name)) {
                this.products.set(mult.name, {
                    x: mult.mult_x,
                    y: mult.mult_y,
                    matrices: new BiVec(this.minDegree),
                    style: {
                        bend: 0,
                        dash: '',
                        color: 'black',
                        styleObject: null,
                    },
                });
            }
            const product = this.products.get(mult.name);
            const oldMatrices = product.matrices.get(x, y);
            if (JSON.stringify(oldMatrices) === JSON.stringify(mult.matrices)) {
                continue;
            }
            product.matrices.set(x, y, mult.matrices);

            if (this.visibleStructlines.has(mult.name)) {
                for (const [r, page] of this.chart.pages.entries()) {
                    const matrix = ExtSseq.getPage(mult.matrices, r + MIN_PAGE);
                    const oldMatrix = ExtSseq.getPage(
                        oldMatrices,
                        r + MIN_PAGE,
                    );

                    if (JSON.stringify(matrix) === JSON.stringify(oldMatrix)) {
                        continue;
                    }

                    if (oldMatrices !== undefined) {
                        this.clearOld(`structline-${mult.name}`, x, y, page);
                    }

                    for (const line of ExtSseq.drawMatrix(
                        matrix,
                        x,
                        x + mult.mult_x,
                        y,
                        y + mult.mult_y,
                        product.style.bend,
                    )) {
                        line.classList.add(`structline`);
                        line.classList.add(`structline-${mult.name}`);
                        line.classList.add(`structline-${mult.name}-${x}-${y}`);
                        // Go under classes
                        page.insertBefore(line, page.firstChild);
                    }
                }
            }
        }
        if (this.hasSelected(x, y)) {
            this.refreshPanel?.();
        }
    }

    /**
     * @description Hides a structline with a given name by removing it from the
     * `visibleStructlines` collection and from the DOM tree, if it exists. If the
     * structline is not visible initially, the method does nothing.
     * 
     * @param {string} name - Used to identify struct lines to be hidden.
     */
    hideStructlines(name) {
        if (!this.visibleStructlines.has(name)) {
            return;
        }
        this.visibleStructlines.delete(name);
        this.chart.shadowRoot
            .querySelectorAll(`.structline-${CSS.escape(name)}`)
            .forEach(x => x.remove());
    }

    /**
     * @description Makes certain lines visible and inserts them into an HTML page. The
     * lines are derived from matrices based on product information, with their position
     * and style determined by the product's mult object.
     * 
     * @param {string} name - Used to identify specific structural lines.
     */
    showStructlines(name) {
        if (this.visibleStructlines.has(name)) {
            return;
        }
        this.visibleStructlines.add(name);
        const mult = this.products.get(name);
        const matrices = mult.matrices;
        for (const [x_, row] of matrices.data.entries()) {
            const x = x_ + this.minDegree;
            for (const [y, pageMatrices] of row.entries()) {
                if (pageMatrices === undefined) {
                    continue;
                }
                for (const [r, page] of this.chart.pages.entries()) {
                    const pageIdx = Math.min(pageMatrices.length - 1, r);
                    const matrix = pageMatrices[pageIdx];

                    for (const line of ExtSseq.drawMatrix(
                        matrix,
                        x,
                        x + mult.x,
                        y,
                        y + mult.y,
                        mult.style.bend,
                    )) {
                        line.classList.add(`structline`);
                        line.classList.add(`structline-${name}`);
                        line.classList.add(`structline-${name}-${x}-${y}`);
                        // Go under classes
                        page.insertBefore(line, page.firstChild);
                    }
                }
            }
        }
    }

    /**
     * Sort the nodes in the svg in some consistent way. This is useful for
     * integration testing.
     */
    sort() {
        /**
         * @description Compares two input values, `a` and `b`, and returns an integer value
         * indicating their relative magnitude. It returns 0 if they are equal, 1 if `a` is
         * greater than `b`, and -1 if `a` is less than `b`.
         * 
         * @param {number} a - Compared with the `b` parameter.
         * 
         * @param {number} b - Another value to be compared with `a`.
         * 
         * @returns {number} 0 if a and b are equal, 1 if a is greater than b, and -1 if a
         * is less than b.
         */
        const compare = (a, b) => {
            if (a == b) {
                return 0;
            } else if (a > b) {
                return 1;
            } else {
                return -1;
            }
        };

        for (const page of this.chart.pages) {
            [...page.querySelectorAll('.structline')]
                .sort((a, b) =>
                    compare(a.className.baseVal, b.className.baseVal),
                )
                .forEach(n => page.appendChild(n));

            [...page.querySelectorAll('.differential')]
                .sort((a, b) =>
                    compare(a.className.baseVal, b.className.baseVal),
                )
                .forEach(n => page.appendChild(n));

            [...page.querySelectorAll('.class-group')]
                .sort((a, b) =>
                    compare(a.className.baseVal, b.className.baseVal),
                )
                .forEach(n => page.appendChild(n));
        }
    }

    /**
     * @description Retrieves and returns a differential value from a cache stored as a
     * nested structure. The cache is accessed using the keys `x` and `y`, with the `page`
     * parameter specifying which page in the nested structure to retrieve from.
     * 
     * @param {string} x - Used as a key.
     * 
     * @param {number} y - Used for indexing.
     * 
     * @param {number} page - 1-indexed.
     * 
     * @returns {any} Retrieved from a nested data structure within a map named
     * `differentials`. The returned value corresponds to a specific key-value pair in
     * the nested structure and is indexed by subtracting a constant `MIN_PAGE` from the
     * given page number.
     */
    getDifferentials(x, y, page) {
        return this.differentials.get(x, y)?.[page - MIN_PAGE];
    }

    /**
     * @description Determines whether there are any classes found for a given combination
     * of parameters `x`, `y`, and `page`. It returns a boolean value indicating whether
     * classes exist and are not empty.
     * 
     * @param {number} x - Used for computation.
     * 
     * @param {number} y - Irrelevant for the function's operation.
     * 
     * @param {object} page - Used to identify classes within a web page.
     * 
     * @returns {boolean} True if the length of the array returned by `this.getClasses(x,
     * y, page)` is greater than zero and false otherwise.
     */
    hasClasses(x, y, page) {
        const classes = this.getClasses(x, y, page);
        return classes !== undefined && classes.length > 0;
    }

    /**
     * @description Retrieves a page from an array-like object (`v`) based on the given
     * row index (`r`). If `v` is undefined, it returns undefined. Otherwise, it adjusts
     * `r` to ensure it's within the valid range and returns the element at that index.
     * 
     * @param {(string | number)[]} v - An array to be retrieved from.
     * 
     * @param {number} r - Used to retrieve an element from an array `v`.
     * 
     * @returns {v[r]} One element from an array v at index r, where r is adjusted according
     * to certain conditions. If the returned value is undefined, it means that the array
     * is empty. Otherwise, it is a specific element from the array.
     */
    static getPage(v, r) {
        if (v === undefined) {
            return undefined;
        }
        r -= MIN_PAGE;
        if (r >= v.length) r = v.length - 1;
        return v[r];
    }

    /**
     * Get the list of classes on a given page.
     */
    getClasses(x, y, page) {
        return ExtSseq.getPage(this.classes.get(x, y), page);
    }

    /**
     * @description Adds a 'highlight' class to all elements with the class name
     * `.class-group-${x}-${y}` within the shadow root of a chart, where `x` and `y` are
     * parameters passed to the method.
     * 
     * @param {number} x - Used to identify a class group.
     * 
     * @param {string} y - Used to identify specific elements.
     */
    highlightClass(x, y) {
        this.chart.shadowRoot
            .querySelectorAll(`.class-group-${x}-${y}`)
            .forEach(x => x.classList.add('highlight'));
    }

    /**
     * @description Removes the "highlight" class from all elements with the class name
     * ".highlight" within its chart's shadow root subtree, effectively clearing any
     * previous highlighting.
     */
    clearHighlight() {
        this.chart.shadowRoot
            .querySelectorAll(`.highlight`)
            .forEach(x => x.classList.remove('highlight'));
    }
}
