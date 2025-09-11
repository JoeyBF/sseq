const { test, expect } = require('@playwright/test');
const { TestUtils } = require('./utils');

test.describe('Panel Tests', () => {
  let testUtils;

  test.beforeEach(async ({ page }) => {
    testUtils = new TestUtils(page, test.info().config);
    await testUtils.go('/?module=tmf2');
    await testUtils.waitComplete();
  });

  test('should rotate panels', async ({ page }) => {
    const panel = await testUtils.getPanel();
    const h2Text = await page.evaluate((panel) => panel.querySelector('h2').textContent, panel);
    expect(h2Text).toBe('Vanishing line');

    // Navigate to history panel
    await page.keyboard.press('j');
    const historyPanelElements = await page.evaluate((panel) => panel.querySelectorAll('*').length, await testUtils.getPanel());
    expect(historyPanelElements).toBe(0);

    // Back to Main
    await page.keyboard.press('j');
    await page.keyboard.press('j');
    const mainPanelH2 = await page.evaluate((panel) => panel.querySelector('h2').textContent, await testUtils.getPanel());
    expect(mainPanelH2).toBe('Vanishing line');

    // Now to Prod panel
    await page.keyboard.press('k');
    const prodPanelDetails = await page.evaluate((panel) => panel.querySelectorAll('details').length, await testUtils.getPanel());
    expect(prodPanelDetails).toBe(3);
  });

  test('should handle structline styling', async ({ page }) => {
    await testUtils.selectPanel('Prod');

    const panel = await testUtils.getPanel();
    const details = await page.evaluate((panel) => Array.from(panel.querySelectorAll('details')), panel);

    // Open first two details
    await page.evaluate((details) => {
      details[0].click();
      details[1].click();
    }, details);

    // Set color in first detail
    const colorInputRow = await page.locator('details').first().locator('input-row[label="Color"]');
    const colorInput = colorInputRow.locator('input');
    await colorInput.clear();
    await colorInput.fill('red');

    // Set bend in second detail
    const bendInputRow = await page.locator('details').nth(1).locator('input-row[label="Bend"]');
    const bendInput = bendInputRow.locator('input');
    await bendInput.clear();
    await bendInput.fill('20');

    // Set dash in second detail
    const dashInputRow = await page.locator('details').nth(1).locator('input-row[label="Dash"]');
    const dashInput = dashInputRow.locator('input');
    await dashInput.clear();
    await dashInput.fill('0.05,0.05');

    // Click third checkbox
    await page.locator('div > checkbox-switch').nth(2).click();

    await testUtils.checkSvg('tmf_structline_style.svg');
  });
});