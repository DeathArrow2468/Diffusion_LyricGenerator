/* Run against the frontend on port 5173. Requires Playwright and Chromium. */
const assert = require("node:assert/strict");
const path = require("node:path");
const { chromium } = require("playwright");

(async () => {
  const browser = await chromium.launch({
    headless: true,
    ...(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE
      ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE }
      : {}),
  });
  const context = await browser.newContext({
    viewport: { width: 1440, height: 1100 },
  });
  const page = await context.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  const base = process.env.LYRICWALA_TEST_URL || "http://127.0.0.1:5173";
  const shot = async (name) => {
    if (process.env.LYRICWALA_SCREENSHOTS)
      await page.screenshot({
        path: path.join(process.env.LYRICWALA_SCREENSHOTS, name + ".png"),
        fullPage: true,
      });
  };
  const visible = async (selector) =>
    assert.equal(await page.locator(selector).isVisible(), true, selector);
  const noOverflow = async () =>
    assert.equal(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
      true,
      "horizontal overflow",
    );
  try {
    await page.goto(base);
    assert.equal(await page.title(), "LyricWala — Your next song starts here");
    assert.equal(await page.locator("#genre option").count(), 7);
    await noOverflow();
    await shot("light");
    await page.click("#recordButton");
    assert.equal(
      await page.locator("#recordButton").getAttribute("aria-pressed"),
      "true",
    );
    await page.click("#darkTheme");
    assert.equal(await page.locator("html").getAttribute("data-theme"), "dark");
    await shot("dark");
    await page.reload();
    assert.equal(await page.locator("html").getAttribute("data-theme"), "dark");
    await page.click("#helpButton");
    await visible("#helpDialog");
    await page.keyboard.press("Escape");
    await page.click("#lightTheme");
    await page.click('[data-preset="0"]');
    assert.ok((await page.inputValue("#prompt")).length > 20);
    await page.selectOption("#genre", "R&B");
    await page.click('[data-mood="Romantic"]');
    await page.locator("#creativity").evaluate((el) => {
      el.value = "80";
      el.dispatchEvent(new Event("input", { bubbles: true }));
    });
    assert.equal(
      await page.textContent("#creativityLabel"),
      "Beautifully unexpected",
    );
    await page.click("#generateButton");
    await visible("#loadingPanel");
    await page.waitForFunction(
      () =>
        Number(
          document
            .querySelector("#progressTrack")
            .getAttribute("aria-valuenow"),
        ) > 0,
    );
    await shot("loading");
    await page.click("#cancelButton");
    await page.waitForFunction(
      () => !document.querySelector("#generateButton").disabled,
    );
    assert.ok((await page.inputValue("#prompt")).length > 20);
    await page.click("#generateButton");
    await page
      .locator("#lyricsState")
      .waitFor({ state: "visible", timeout: 15000 });
    await visible("#sampleNotice");
    await shot("result");
    await page.click("#saveButton");
    assert.equal(await page.textContent("#savedCount"), "1");
    const downloaded = page.waitForEvent("download");
    await page.click("#downloadButton");
    assert.ok((await downloaded).suggestedFilename().endsWith(".txt"));
    await page.click("#navSaved");
    assert.equal(await page.locator(".saved-card").count(), 1);
    await page.fill("#savedSearch", "unmatchable-words");
    assert.equal(await page.locator(".saved-card").count(), 0);
    await page.fill("#savedSearch", "");
    await page.locator(".saved-card .text-button").click();
    await visible("#lyricsState");
    await page.reload();
    assert.equal(await page.textContent("#savedCount"), "1");
    await page.click("#navSaved");
    await page.locator(".saved-card .icon-button").click();
    assert.equal(await page.textContent("#savedCount"), "0");
    await page.click("#navStudio");
    for (const width of [375, 320, 768, 1024]) {
      await page.setViewportSize({ width, height: 900 });
      await noOverflow();
      if (width === 375) await shot("mobile");
    }
    await page.setViewportSize({ width: 1440, height: 1100 });
    await page.emulateMedia({ reducedMotion: "reduce" });
    assert.equal(
      await page
        .locator("#recordDisc")
        .evaluate((el) => getComputedStyle(el).animationName),
      "none",
    );

    // An unavailable API must show an error, never silently substitute demo text.
    await page.addInitScript(() => {
      window.LYRICWALA_CONFIG = {
        mode: "api",
        apiBaseUrl: "http://127.0.0.1:8000",
      };
    });
    await page.route("**/api/**", (route) => route.abort());
    await page.reload();
    await page.fill("#prompt", "A story about coming home");
    await page.click("#generateButton");
    await page.locator("#errorBox").waitFor({ state: "visible" });
    assert.equal(await page.inputValue("#prompt"), "A story about coming home");
    await page.unroute("**/api/**");

    if (process.env.LYRICWALA_TEST_API) {
      await page.addInitScript((api) => {
        window.LYRICWALA_CONFIG = { mode: "api", apiBaseUrl: api };
      }, process.env.LYRICWALA_TEST_API);
      await page.reload();
      await page.fill("#prompt", "A road towards a new beginning");
      await page.click("#generateButton");
      await page.waitForFunction(
        () =>
          Number(
            document
              .querySelector("#progressTrack")
              .getAttribute("aria-valuenow"),
          ) > 0,
      );
      await page
        .locator("#lyricsState")
        .waitFor({ state: "visible", timeout: 15000 });
      await visible("#sampleNotice");
    }
    assert.deepEqual(errors, []);
    console.log(
      "UI checks passed: themes, record, presets, controls, progress, cancel, result, save, search, delete, download, persistence, responsive widths, reduced motion, API errors" +
        (process.env.LYRICWALA_TEST_API ? ", live API streaming." : "."),
    );
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
