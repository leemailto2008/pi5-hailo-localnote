/**
 * LocalNote 聽記盒 - Service Worker
 * Version: localnote-v1 (Task 10: 漸進式 Web 應用程式離線快取)
 */

const CACHE_NAME = 'localnote-cache-v1';
const PRECACHE_RESOURCES = [
  '/',
  '/static/index.html',
  '/static/manifest.json',
  '/static/icon-192.png',
  '/static/icon-512.png',
  '/static/wifi.png'
];

// 安裝階段：預快取核心外殼資源 (Shell Resources)
self.addEventListener('install', (event) => {
  self.skipWaiting();
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log('[SW] Pre-caching offline shell resources');
      return cache.addAll(PRECACHE_RESOURCES);
    }).catch((err) => {
      console.warn('[SW] Pre-cache warning:', err);
    })
  );
});

// 啟動階段：清理舊版本快取並立即接管控制權
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(
        keys.map((key) => {
          if (key !== CACHE_NAME) {
            console.log('[SW] Removing deprecated cache:', key);
            return caches.delete(key);
          }
        })
      );
    }).then(() => self.clients.claim())
  );
});

// 攔截請求：API/WebSocket 走網路，靜態資源採 Stale-While-Revalidate 與離線保底
self.addEventListener('fetch', (event) => {
  const req = event.request;
  const url = new URL(req.url);

  // 非同源或非 GET 請求一律略過
  if (req.method !== 'GET' || !url.origin.startsWith(self.location.origin)) {
    return;
  }

  // 1. API 端點、音訊播放快取、WebSocket 握手一律不快取 (Network-Only)
  if (url.pathname.startsWith('/api/') || url.pathname.startsWith('/audio/') || url.pathname.startsWith('/ws/')) {
    return;
  }

  // 2. 靜態資源與 HTML 導覽：Stale-While-Revalidate + Cache Fallback
  event.respondWith(
    caches.match(req).then((cachedResponse) => {
      const fetchPromise = fetch(req).then((networkResponse) => {
        if (networkResponse && networkResponse.status === 200 && networkResponse.type === 'basic') {
          const responseToCache = networkResponse.clone();
          caches.open(CACHE_NAME).then((cache) => {
            cache.put(req, responseToCache);
          });
        }
        return networkResponse;
      }).catch((fetchErr) => {
        // 離網完全無網路時，若導覽頁面未在快取中，保底回傳主頁面
        if (req.mode === 'navigate') {
          return caches.match('/') || caches.match('/static/index.html');
        }
        throw fetchErr;
      });

      // 快取命中則瞬間回傳快取，背景自動同步最新版本
      return cachedResponse || fetchPromise;
    })
  );
});
