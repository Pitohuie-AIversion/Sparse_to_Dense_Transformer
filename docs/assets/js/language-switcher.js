// Enhanced language switching for runtime content switching
document.addEventListener('DOMContentLoaded', function() {
  var currentPath = window.location.pathname;
  var baseUrl = document.querySelector('meta[name="site-baseurl"]')?.getAttribute('content') || '';
  var isZh = currentPath.includes('/zh/') && !currentPath.endsWith('/') && !currentPath.endsWith('/index.html');
  var englishBtn = document.querySelector('.lang-btn-en');
  var chineseBtn = document.querySelector('.lang-btn-zh');
  
  if (!englishBtn || !chineseBtn) {
    return; // Exit if buttons are not found
  }
  
  // Function to switch language content on the same page
  function switchLanguageContent(lang) {
    // Select all language content blocks with standardized markup
    var allContent = document.querySelectorAll('.lang-content[data-lang-zh], .lang-content[data-lang-en]');
    allContent.forEach(function(element) {
      var elementLang = element.hasAttribute('data-lang-zh') ? 'zh' : 'en';
      if (elementLang === lang) {
        element.style.display = 'block';
      } else {
        element.style.display = 'none';
      }
    });
    
    // Update active button state
    if (lang === 'zh') {
      englishBtn.classList.remove('active');
      chineseBtn.classList.add('active');
    } else {
      englishBtn.classList.add('active');
      chineseBtn.classList.remove('active');
    }
    
    // Dispatch custom event for other components
    window.dispatchEvent(new CustomEvent('languageChanged', {
      detail: { language: lang }
    }));
  }
  
  // Set up click handlers for runtime switching on index page
  if (currentPath === baseUrl + '/' || currentPath === baseUrl || currentPath.endsWith('/index.html')) {
    englishBtn.onclick = function(e) {
      e.preventDefault();
      switchLanguageContent('en');
    };
    chineseBtn.onclick = function(e) {
      e.preventDefault();
      switchLanguageContent('zh');
    };
    
    // Initialize with detected language
    switchLanguageContent(isZh ? 'zh' : 'en');
  } else {
    // For other pages, use navigation-based switching
    if (isZh) {
      var enPath = currentPath.replace('/zh/', '/').replace('/zh', '');
      if (enPath === baseUrl || enPath === baseUrl + '/') {
        enPath = baseUrl + '/';
      }
      englishBtn.onclick = function() {
        window.location.href = enPath;
      };
    } else {
      var zhPath;
      if (currentPath === baseUrl + '/' || currentPath === baseUrl) {
        zhPath = baseUrl + '/zh/';
      } else {
        zhPath = currentPath.replace(baseUrl + '/', baseUrl + '/zh/');
      }
      chineseBtn.onclick = function() {
        window.location.href = zhPath;
      };
    }
  }
});