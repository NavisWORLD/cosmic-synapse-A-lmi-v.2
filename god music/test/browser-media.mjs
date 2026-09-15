import assert from 'node:assert/strict';
import { chromium } from 'playwright';

const browser = await chromium.launch({
  headless: true,
  args: [
    '--use-fake-device-for-media-stream',
    '--use-fake-ui-for-media-stream',
  ],
});

const context = await browser.newContext({
  permissions: ['microphone'],
});
const page = await context.newPage();
const pageErrors = [];
page.on('pageerror', error => pageErrors.push(error.message));

try {
  await page.goto('http://127.0.0.1:4173', { waitUntil: 'networkidle' });
  await page.waitForFunction(() => Boolean(window.conductor?.audioEngine?.isInitialized));

  await page.click('#calibrateBtn');
  await page.waitForFunction(() => {
    const stream = window.conductor?.microphoneStream;
    return Boolean(stream && stream.getAudioTracks().some(track => track.readyState === 'live'));
  });

  const liveState = await page.evaluate(() => {
    const stream = window.conductor.microphoneStream;
    window.__vmMicTrack = stream.getAudioTracks()[0];
    return {
      streamPresent: Boolean(stream),
      audioTracks: stream.getAudioTracks().length,
      readyState: window.__vmMicTrack.readyState,
    };
  });
  assert.equal(liveState.streamPresent, true);
  assert.ok(liveState.audioTracks > 0);
  assert.equal(liveState.readyState, 'live');

  await page.evaluate(() => window.conductor.stopBand());
  await page.waitForTimeout(100);

  const stoppedState = await page.evaluate(() => ({
    streamCleared: window.conductor.microphoneStream === null,
    previousTrackState: window.__vmMicTrack.readyState,
    audioEngineMicrophoneCleared: window.conductor.audioEngine.microphone === null,
  }));

  assert.equal(stoppedState.streamCleared, true, 'stop must clear microphoneStream');
  assert.equal(stoppedState.previousTrackState, 'ended', 'stop must end microphone tracks');
  assert.equal(
    stoppedState.audioEngineMicrophoneCleared,
    true,
    'stop must disconnect and clear the Web Audio microphone source',
  );
  assert.deepEqual(pageErrors, [], `page errors: ${pageErrors.join('; ')}`);

  console.log(JSON.stringify({
    gate: 'browser-fake-media',
    classification: 'VERIFIED_SIMULATED_DEVICE_PATH',
    liveState,
    stoppedState,
  }));
} finally {
  await context.close();
  await browser.close();
}
