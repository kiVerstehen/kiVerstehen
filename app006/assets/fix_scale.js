window.addEventListener('load', function () {
    const content = document.getElementById('_dash-app-content');
    if (content) {
        content.style.transform = 'scale(0.8)';
        content.style.transformOrigin = '0 0';
        content.style.width = '125%';
        content.style.height = '125%';
    }
});