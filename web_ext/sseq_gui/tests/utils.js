const fs = require('fs');
const path = require('path');

const SVGNS = "http://www.w3.org/2000/svg";

class TestUtils {
  constructor(page, testConfig) {
    this.page = page;
    this.testConfig = testConfig;
    this.tempdir = testConfig.projects[0].tempdir || '/tmp';
  }

  async go(path) {
    await this.page.goto(`http://localhost:8080${path}`);
  }

  async waitComplete(timeout = 10000) {
    // If the commands we send out are done via a callback, then they might
    // not have been sent out yet when we call wait_complete. Sleep for a
    // very small amount of time to ensure these callbacks have been
    // handled.
    await this.page.waitForTimeout(10);
    await this.page.waitForFunction(
      () => window.display !== undefined && window.display.runningSign.style.display == 'none',
      { timeout }
    );
  }

  async unitSvg() {
    return await this.page.evaluate(() => window.unitSseq.chart.svg);
  }

  async mainSvg() {
    return await this.page.evaluate(() => window.mainSseq.chart.svg);
  }

  async checkFile(filename, value) {
    const fullPath = path.join(__dirname, 'benchmarks', filename);

    if (process.env.UPDATE_BENCHMARKS || this.testConfig.updateSnapshots) {
      fs.writeFileSync(fullPath, value);
      return;
    }

    let benchmark;
    try {
      benchmark = fs.readFileSync(fullPath, 'utf8');
    } catch (err) {
      fs.writeFileSync(fullPath, value);
      return;
    }

    const equal = filename.endsWith('.svg')
      ? this.cleanSvg(benchmark) === this.cleanSvg(value)
      : benchmark === value;

    if (!equal) {
      const newPath = fullPath.replace(/(\.[^.]+)$/, '-new$1');
      fs.writeFileSync(newPath, value);
      throw new Error(`${path.basename(fullPath)} changed. New version saved at ${path.basename(newPath)}`);
    }
  }

  async checkSvg(filename) {
    await this.page.evaluate(() => window.mainSseq.sort());
    const svg = await this.page.locator('#mainSvg').getAttribute('outerHTML');
    await this.checkFile(filename, svg);
  }

  async checkPages(suffix, maxPage) {
    await this.page.locator('#mainSvg').click();
    await this.waitComplete();

    for (let page = 2; page <= maxPage; page++) {
      await this.checkSvg(`${suffix}_e${page}.svg`);
      await this.page.keyboard.press('ArrowRight');
    }

    for (let page = 2; page <= maxPage; page++) {
      await this.page.keyboard.press('ArrowLeft');
    }
  }

  async clickClass(x, y, main = true) {
    const svgSelector = main ? '#mainSvg' : '#unitSvg';
    await this.page.locator(`${svgSelector} g [data-x='${x}'][data-y='${y}'] > circle`).click();
  }

  async selectPanel(name) {
    const panelLinks = await this.page.evaluate(() => {
      const head = window.display.currentPanel.head;
      return Array.from(head.querySelectorAll('a')).map(a => ({ text: a.textContent, element: a }));
    });

    const found = panelLinks.find(link => link.text === name);
    if (!found) {
      throw new Error(`Panel ${name} not found`);
    }

    await this.page.evaluate((element) => element.click(), found.element);
  }

  async getPanel() {
    return await this.page.evaluate(() => window.display.currentPanel.inner);
  }

  async getSidebar() {
    return await this.page.evaluate(() => window.display.sidebar);
  }

  async clickButton(text) {
    const button = this.page.locator(`button:has-text("${text}")`);
    await button.click();
  }

  async zoomOut(sseq = "main") {
    await this.page.evaluate((sseqName) => {
      window[`${sseqName}Sseq`].chart.svg.dispatchEvent(
        new WheelEvent("wheel", {
          view: window,
          bubbles: true,
          cancelable: true,
          clientX: 300,
          clientY: 300,
          deltaY: 10000,
        })
      );
    }, sseq);
  }

  cleanSvg(svg) {
    // This is a simplified version of the Python XML cleaning
    // In a real implementation, you'd want to use a proper XML parser
    // For now, we'll do basic string cleaning
    let cleaned = svg.replace(/ style=""/g, '');

    // Remove dynamic attributes that change between runs
    cleaned = cleaned.replace(/viewBox="[^"]*"/g, '');
    cleaned = cleaned.replace(/transform="[^"]*"/g, '');

    // Remove elements that contain dynamic data
    cleaned = cleaned.replace(/<g[^>]*id="axisLabels"[^>]*>.*?<\/g>/gs, '');
    cleaned = cleaned.replace(/<rect[^>]*id="xBlock"[^>]*\/>/gs, '');
    cleaned = cleaned.replace(/<rect[^>]*id="yBlock"[^>]*\/>/gs, '');
    cleaned = cleaned.replace(/<path[^>]*id="axis"[^>]*\/>/gs, '');

    return cleaned;
  }
}

module.exports = { TestUtils };