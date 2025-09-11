const { test, expect } = require('@playwright/test');
const { TestUtils } = require('./utils');

test.describe('C3 and Calpha Tests', () => {
  let testUtils;

  test.beforeEach(async ({ page }) => {
    testUtils = new TestUtils(page, test.info().config);
  });

  test('should handle C3 differentials', async ({ page }) => {
    await testUtils.go('/?module=C3&degree=36');
    await testUtils.waitComplete();

    await testUtils.clickClass(18, 2);
    await page.keyboard.press('d');
    await testUtils.clickClass(17, 4);
    await page.keyboard.press('Enter');

    await testUtils.clickClass(19, 2);
    await page.keyboard.press('d');
    await testUtils.clickClass(18, 4);
    await page.keyboard.press('Tab');
    await page.keyboard.type('[2]');
    await page.keyboard.press('Enter');

    // Differential propagation checks that v₁ and β products are working
    await testUtils.checkPages('C3_differential', 3);
  });

  test('should handle Calpha differentials', async ({ page }) => {
    await testUtils.go('/?module=Calpha&degree=36');
    await testUtils.waitComplete();

    await testUtils.clickClass(0, 0);
    await page.keyboard.press('p');

    const mainSvg = await testUtils.mainSvg();
    await page.evaluate((svg) => svg.click(), mainSvg);
    await testUtils.selectPanel('Prod');

    await testUtils.clickButton('Add');
    await testUtils.clickButton('Show more');

    await page.keyboard.type('20');
    await page.keyboard.press('Enter');
    await testUtils.waitComplete();

    await testUtils.zoomOut('unit');
    await testUtils.clickClass(18, 2, false);
    await testUtils.clickButton('Add differential');
    await testUtils.clickClass(17, 4, false);
    await page.keyboard.press('Tab');
    await page.keyboard.press('Tab');
    await page.keyboard.type('g_2');
    await page.keyboard.press('Tab');
    await page.keyboard.type('g_1b');
    await page.keyboard.press('Enter');
    await page.keyboard.press('Escape');

    // Differential propagation checks that v₁ and β products are working
    await testUtils.checkPages('Calpha_differential', 3);
  });
});