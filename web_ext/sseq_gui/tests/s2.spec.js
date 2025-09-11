const { test, expect } = require('@playwright/test');
const { TestUtils } = require('./utils');
const path = require('path');
const fs = require('fs');

test.describe('S_2 Tests', () => {
  let testUtils;

  test.beforeEach(async ({ page }) => {
    testUtils = new TestUtils(page, test.info().config);
    await testUtils.go('/?module=S_2&degree=20');
    await testUtils.waitComplete();
    await testUtils.zoomOut();
  });

  test('should handle differentials', async ({ page }) => {
    await testUtils.clickClass(15, 1);
    await page.keyboard.press('d');
    await testUtils.clickClass(14, 3);
    await page.keyboard.press('Enter');

    await testUtils.clickClass(15, 2);
    await page.keyboard.press('d');
    await testUtils.clickClass(14, 5);
    await page.keyboard.press('Enter');

    await testUtils.clickClass(17, 4);
    await page.keyboard.press('d');
    await testUtils.clickClass(16, 6);
    await page.keyboard.press('Enter');

    await testUtils.clickClass(18, 4);
    await testUtils.selectPanel('Diff');
    await testUtils.clickButton('Add Differential');
    await testUtils.clickClass(17, 6);
    await page.keyboard.type('[0, 1]');
    await page.keyboard.press('Enter');

    await testUtils.checkPages('S_2_differential', 4);
  });

  test('should handle permanent classes', async ({ page }) => {
    await testUtils.clickClass(0, 0);
    await page.keyboard.press('p');

    await testUtils.clickClass(8, 3);
    await testUtils.selectPanel('Diff');
    await testUtils.clickButton('Add Permanent Class');

    await testUtils.checkPages('S_2_permanent', 4);
  });

  test('should resolve further', async ({ page }) => {
    const mainSvg = await testUtils.mainSvg();
    await page.evaluate((svg) => svg.click(), mainSvg);
    await testUtils.clickButton('Resolve further');
    await page.keyboard.type('36');
    await page.keyboard.press('Enter');

    await testUtils.waitComplete();
    await testUtils.zoomOut();

    await testUtils.checkPages('S_2_further', 4);
  });

  test('should handle multiplication', async ({ page }) => {
    await testUtils.clickClass(8, 3);
    await page.keyboard.press('m');
    await page.keyboard.type('c_0');
    await page.keyboard.press('Enter');

    await testUtils.clickClass(9, 5);
    await page.keyboard.press('m');
    await page.keyboard.type('Ph_1');
    await page.keyboard.press('Enter');

    await testUtils.clickClass(14, 4);
    await page.keyboard.press('m');
    await page.keyboard.type('d_0');
    await page.keyboard.press('Enter');

    await testUtils.clickClass(20, 4);
    await testUtils.selectPanel('Main');
    await testUtils.clickButton('Add Product');
    await page.keyboard.type('g');
    await page.keyboard.press('Enter');

    await testUtils.checkPages('S_2_multiplication', 4);
  });

  test('should propagate differentials', async ({ page }) => {
    await testUtils.clickClass(17, 4);
    await testUtils.selectPanel('Diff');
    await page.locator('div.panel-line').first().click();
    await page.keyboard.press('Tab');
    await page.keyboard.press('Tab');
    await page.keyboard.type('e_0');
    await page.keyboard.press('Tab');
    await page.keyboard.type('h_1^2 d_0');
    await page.keyboard.press('Enter');

    await testUtils.clickClass(18, 4);
    await page.locator('div.panel-line').nth(3).click();
    await page.keyboard.press('Tab');
    await page.keyboard.press('Tab');
    await page.keyboard.type('f_0');
    await page.keyboard.press('Tab');
    await page.keyboard.type('h_0^2 e_0');
    await page.keyboard.press('Enter');

    await testUtils.checkPages('S_2_propagate_diff', 4);
  });

  test('should handle undo and redo', async ({ page }) => {
    await testUtils.clickButton('Undo');
    await testUtils.clickButton('Undo');
    await testUtils.checkPages('S_2_multiplication', 4);

    await testUtils.clickButton('Redo');
    await testUtils.clickButton('Redo');
    await testUtils.checkPages('S_2_propagate_diff', 4);
  });

  test('should save and load history', async ({ page }) => {
    await testUtils.clickButton('Save');
    await page.keyboard.type('s_2.save');
    await page.keyboard.press('Enter');

    // Wait for file to be saved
    let timeout = 100;
    const savePath = path.join(testUtils.tempdir, 's_2.save');
    let fileContents;

    while (timeout <= 10000) {
      await page.waitForTimeout(timeout);
      try {
        fileContents = fs.readFileSync(savePath, 'utf8');
        if (fileContents) break;
      } catch (err) {
        // File doesn't exist yet
      }
      timeout *= 2;
    }

    if (!fileContents) {
      throw new Error('Save file was not created within timeout');
    }

    await testUtils.checkFile('s_2.save', fileContents);

    await testUtils.go('/');
    await page.locator('#history-upload').setInputFiles(savePath);
    await testUtils.waitComplete();

    await testUtils.checkPages('S_2_propagate_diff', 4);
  });
});