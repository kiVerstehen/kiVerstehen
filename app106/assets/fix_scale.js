function applyScaling() {
  const content = document.getElementById('_dash-app-content');
  if (content) {
    content.style.transform = 'scale(0.7)';
    content.style.transformOrigin = '0 0';
    content.style.width = '125%';
    content.style.height = '125%';
    document.body.style.overflowX = 'hidden';
    document.documentElement.style.overflowX = 'hidden';
    return true;
  }
  return false;
}

// Versuche direkt anwenden, falls schon da
if (!applyScaling()) {
  // Beobachte DOM-Änderungen und wende Styling an sobald vorhanden
  const observer = new MutationObserver(() => {
    if (applyScaling()) {
      observer.disconnect();
    }
  });

  observer.observe(document.body, { childList: true, subtree: true });
}