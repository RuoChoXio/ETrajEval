window.HELP_IMPROVE_VIDEOJS = false;

// More Works Dropdown Functionality
function toggleMoreWorks() {
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');
    
    if (dropdown.classList.contains('show')) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    } else {
        dropdown.classList.add('show');
        button.classList.add('active');
    }
}

// Close dropdown when clicking outside
document.addEventListener('click', function(event) {
    const container = document.querySelector('.more-works-container');
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');
    
    if (container && !container.contains(event.target)) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    }
});

// Close dropdown on escape key
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        const dropdown = document.getElementById('moreWorksDropdown');
        const button = document.querySelector('.more-works-btn');
        dropdown.classList.remove('show');
        button.classList.remove('active');
    }
});

// Copy BibTeX to clipboard
function copyBibTeX() {
    const bibtexElement = document.getElementById('bibtex-code');
    const button = document.querySelector('.copy-bibtex-btn');
    const copyText = button.querySelector('.copy-text');
    
    if (bibtexElement) {
        navigator.clipboard.writeText(bibtexElement.textContent).then(function() {
            // Success feedback
            button.classList.add('copied');
            copyText.textContent = 'Cop';
            
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        }).catch(function(err) {
            console.error('Failed to copy: ', err);
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = bibtexElement.textContent;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            
            button.classList.add('copied');
            copyText.textContent = 'Cop';
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        });
    }
}

// Scroll to top functionality
function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

// Show/hide scroll to top button
window.addEventListener('scroll', function() {
    const scrollButton = document.querySelector('.scroll-to-top');
    if (window.pageYOffset > 300) {
        scrollButton.classList.add('visible');
    } else {
        scrollButton.classList.remove('visible');
    }
});

// Video carousel autoplay when in view
function setupVideoCarouselAutoplay() {
    const carouselVideos = document.querySelectorAll('.results-carousel video');
    
    if (carouselVideos.length === 0) return;
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            const video = entry.target;
            if (entry.isIntersecting) {
                // Video is in view, play it
                video.play().catch(e => {
                    // Autoplay failed, probably due to browser policy
                    console.log('Autoplay prevented:', e);
                });
            } else {
                // Video is out of view, pause it
                video.pause();
            }
        });
    }, {
        threshold: 0.5 // Trigger when 50% of the video is visible
    });
    
    carouselVideos.forEach(video => {
        observer.observe(video);
    });
}

$(document).ready(function() {
    // Check for click events on the navbar burger icon

    var options = {
		slidesToScroll: 1,
		slidesToShow: 1,
		loop: true,
		infinite: true,
		autoplay: true,
		autoplaySpeed: 10000,
    }

	// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);
	
    bulmaSlider.attach();
    
    // Setup video autoplay for carousel
    setupVideoCarouselAutoplay();

})

document.addEventListener('DOMContentLoaded', () => {
  const tabs = document.querySelectorAll('.tabs li');
  const tabContent = document.querySelectorAll('#metric-tab-content .tab-pane');

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      // Get the target tab ID from data-tab attribute
      const targetId = tab.dataset.tab;
      const targetPane = document.getElementById(targetId);

      // Remove 'is-active' from all tabs and panes
      tabs.forEach(t => t.classList.remove('is-active'));
      tabContent.forEach(pane => pane.classList.remove('is-active'));

      // Activate the clicked tab and its corresponding pane
      tab.classList.add('is-active');
      if (targetPane) {
        targetPane.classList.add('is-active');
      }
    });
  });
});


document.addEventListener('DOMContentLoaded', () => {
  const carouselContainer = document.querySelector('#feature-carousel');
  if (carouselContainer) {
    const images = carouselContainer.querySelectorAll('.carousel-image');
    const dots = carouselContainer.querySelectorAll('.carousel-dot');
    const captions = carouselContainer.querySelectorAll('.carousel-caption');
    const intervalTime = 10000; // Time in milliseconds (e.g., 5 seconds)
    let currentIndex = 0;
    let slideInterval;

    function switchToSlide(index) {
      // Deactivate all
      images.forEach(img => img.classList.remove('is-active'));
      dots.forEach(dot => dot.classList.remove('is-active'));
      captions.forEach(cap => cap.classList.remove('is-active'));

      // Activate the target slide
      if (images[index]) images[index].classList.add('is-active');
      if (dots[index]) dots[index].classList.add('is-active');
      if (captions[index]) captions[index].classList.add('is-active');
      
      currentIndex = index;
    }

    function startSlideShow() {
      slideInterval = setInterval(() => {
        const nextIndex = (currentIndex + 1) % images.length;
        switchToSlide(nextIndex);
      }, intervalTime);
    }

    function resetSlideShow() {
      clearInterval(slideInterval);
      startSlideShow();
    }

    // Event listeners for dots
    dots.forEach(dot => {
      dot.addEventListener('click', () => {
        const slideIndex = parseInt(dot.dataset.slide, 10);
        switchToSlide(slideIndex);
        resetSlideShow(); // Reset timer on manual interaction
      });
    });

    // Start the automatic slideshow
    startSlideShow();
  }
});

