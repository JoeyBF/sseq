const { test, expect } = require('@playwright/test');
const { TestUtils } = require('./utils');
const path = require('path');

test.describe('Load Tests', () => {
  let testUtils;

  test.beforeEach(async ({ page }) => {
    testUtils = new TestUtils(page, test.info().config);
  });

  const modules = ['S_2', 'S_3', 'C2v14'];

  for (const module of modules) {
    test(`should load ${module}`, async ({ page }) => {
      await testUtils.go('/');
      await page.locator(`a[data="${module}"]`).click();
      await testUtils.waitComplete();
      await testUtils.checkSvg(`${module}_load.svg`);
    });

    test(`should load ${module} from JSON`, async ({ page }) => {
      const jsonPath = path.resolve(__dirname, '../../../ext/steenrod_modules', `${module}.json`);

      await testUtils.go('/');
      await page.locator('#json-upload').setInputFiles(jsonPath);
      await testUtils.waitComplete();
      await testUtils.checkSvg(`${module}_load.svg`);
    });
  }
});