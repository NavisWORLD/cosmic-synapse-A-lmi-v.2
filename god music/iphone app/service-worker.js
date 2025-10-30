/**
 * Service Worker for God Music - Enables offline functionality
 */

const CACHE_NAME = 'god-music-v1.0.0';
const urlsToCache = [
  '../index.html',
  '../src/main.js',
  '../src/styles/main.css',
  '../src/styles/components.css',
  '../src/styles/mobile.css',
  '../src/core/AudioEngine.js',
  '../src/core/BioSignature.js',
  '../src/core/PhiHarmonics.js',
  '../src/core/PsiCalculator.js',
  '../src/analysis/PitchDetector.js',
  '../src/analysis/SpectralAnalyzer.js',
  '../src/analysis/TempoDetector.js',
  '../src/prediction/PredictiveEngine.js',
  '../src/prediction/PhraseTracker.js',
  '../src/prediction/GrooveLock.js',
  '../src/prediction/ChordPredictor.js',
  '../src/instruments/InstrumentBase.js',
  '../src/instruments/Drums.js',
  '../src/instruments/Bass.js',
  '../src/instruments/Guitar.js',
  '../src/instruments/Piano.js',
  '../src/instruments/Strings.js',
  '../src/instruments/Pads.js',
  '../src/audio/Mixer.js',
  '../src/audio/Synthesis.js',
  '../src/ui/Logger.js',
  '../src/ui/Visualizer.js',
  '../src/ui/InstrumentControls.js',
  '../src/utils/constants.js',
  '../src/utils/helpers.js',
  '../src/utils/validation.js'
];

// Install event - cache resources
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => {
        console.log('Service Worker: Caching files');
        return cache.addAll(urlsToCache.map(url => new Request(url, { cache: 'reload' })))
          .catch((error) => {
            console.warn('Service Worker: Some files failed to cache:', error);
          });
      })
  );
  self.skipWaiting();
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (cacheName !== CACHE_NAME) {
            console.log('Service Worker: Deleting old cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    })
  );
  return self.clients.claim();
});

// Fetch event - serve from cache, fallback to network
self.addEventListener('fetch', (event) => {
  // Skip non-GET requests
  if (event.request.method !== 'GET') {
    return;
  }

  event.respondWith(
    caches.match(event.request)
      .then((response) => {
        // Return cached version or fetch from network
        return response || fetch(event.request)
          .then((response) => {
            // Don't cache if not a valid response
            if (!response || response.status !== 200 || response.type !== 'basic') {
              return response;
            }

            // Clone the response
            const responseToCache = response.clone();

            caches.open(CACHE_NAME)
              .then((cache) => {
                cache.put(event.request, responseToCache);
              });

            return response;
          })
          .catch(() => {
            // Return offline page if available
            if (event.request.destination === 'document') {
              return caches.match('../index.html');
            }
          });
      })
  );
});

