// Global JavaScript functions for the DR Prediction System

// Smooth scrolling for anchor links
document.addEventListener("DOMContentLoaded", () => {
  // Smooth scrolling for anchor links
  const links = document.querySelectorAll('a[href^="#"]')
  links.forEach((link) => {
    link.addEventListener("click", function (e) {
      e.preventDefault()
      const target = document.querySelector(this.getAttribute("href"))
      if (target) {
        target.scrollIntoView({
          behavior: "smooth",
          block: "start",
        })
      }
    })
  })

  // Auto-hide flash messages after 5 seconds
  const flashMessages = document.querySelectorAll('[role="alert"]')
  flashMessages.forEach((message) => {
    setTimeout(() => {
      message.style.opacity = "0"
      setTimeout(() => {
        message.remove()
      }, 300)
    }, 5000)
  })

  // Add loading states to forms
  const forms = document.querySelectorAll("form")
  forms.forEach((form) => {
    form.addEventListener("submit", () => {
      const submitBtn = form.querySelector('button[type="submit"]')
      if (submitBtn) {
        submitBtn.disabled = true
        const originalText = submitBtn.innerHTML
        submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Processing...'

        // Re-enable after 10 seconds as fallback
        setTimeout(() => {
          submitBtn.disabled = false
          submitBtn.innerHTML = originalText
        }, 10000)
      }
    })
  })
})

// Utility functions
function showNotification(message, type = "info") {
  const notification = document.createElement("div")
  notification.className = `fixed top-20 right-4 z-50 p-4 rounded-lg shadow-lg max-w-sm ${
    type === "success"
      ? "bg-green-100 text-green-800 border border-green-200"
      : type === "error"
        ? "bg-red-100 text-red-800 border border-red-200"
        : type === "warning"
          ? "bg-yellow-100 text-yellow-800 border border-yellow-200"
          : "bg-blue-100 text-blue-800 border border-blue-200"
  }`

  notification.innerHTML = `
        <div class="flex items-center">
            <i class="fas ${
              type === "success"
                ? "fa-check-circle"
                : type === "error"
                  ? "fa-exclamation-circle"
                  : type === "warning"
                    ? "fa-exclamation-triangle"
                    : "fa-info-circle"
            } mr-2"></i>
            <span>${message}</span>
            <button onclick="this.parentElement.parentElement.remove()" class="ml-2 text-gray-500 hover:text-gray-700">
                <i class="fas fa-times"></i>
            </button>
        </div>
    `

  document.body.appendChild(notification)

  // Auto-remove after 5 seconds
  setTimeout(() => {
    if (notification.parentElement) {
      notification.remove()
    }
  }, 5000)
}

// File upload validation
function validateFileUpload(input, maxSize = 16 * 1024 * 1024) {
  const file = input.files[0]
  if (!file) return false

  // Check file size
  if (file.size > maxSize) {
    showNotification("File size too large. Maximum size is 16MB.", "error")
    input.value = ""
    return false
  }

  // Check file type
  const allowedTypes = ["image/jpeg", "image/jpg", "image/png"]
  if (!allowedTypes.includes(file.type)) {
    showNotification("Invalid file type. Please upload JPG, JPEG, or PNG files.", "error")
    input.value = ""
    return false
  }

  return true
}

// Format confidence percentage
function formatConfidence(confidence) {
  return (confidence * 100).toFixed(1) + "%"
}

// Format date
function formatDate(dateString) {
  const date = new Date(dateString)
  return date.toLocaleDateString() + " " + date.toLocaleTimeString()
}

// Copy to clipboard
function copyToClipboard(text) {
  navigator.clipboard
    .writeText(text)
    .then(() => {
      showNotification("Copied to clipboard!", "success")
    })
    .catch(() => {
      showNotification("Failed to copy to clipboard", "error")
    })
}

// Modal utilities
function openModal(modalId) {
  const modal = document.getElementById(modalId)
  if (modal) {
    modal.classList.remove("hidden")
    modal.classList.add("flex")
    document.body.style.overflow = "hidden"
  }
}

function closeModal(modalId) {
  const modal = document.getElementById(modalId)
  if (modal) {
    modal.classList.add("hidden")
    modal.classList.remove("flex")
    document.body.style.overflow = "auto"
  }
}

// Close modal when clicking outside
document.addEventListener("click", (e) => {
  if (e.target.classList.contains("fixed") && e.target.classList.contains("inset-0")) {
    const modals = document.querySelectorAll(".fixed.inset-0")
    modals.forEach((modal) => {
      if (modal.contains(e.target) && e.target === modal) {
        modal.classList.add("hidden")
        modal.classList.remove("flex")
        document.body.style.overflow = "auto"
      }
    })
  }
})

// Keyboard shortcuts
document.addEventListener("keydown", (e) => {
  // Close modals with Escape key
  if (e.key === "Escape") {
    const visibleModals = document.querySelectorAll(".fixed.inset-0.flex")
    visibleModals.forEach((modal) => {
      modal.classList.add("hidden")
      modal.classList.remove("flex")
      document.body.style.overflow = "auto"
    })
  }
})

// Progress bar animation
function animateProgressBar(elementId, targetPercent, duration = 1000) {
  const element = document.getElementById(elementId)
  if (!element) return

  let currentPercent = 0
  const increment = targetPercent / (duration / 16) // 60fps

  const timer = setInterval(() => {
    currentPercent += increment
    if (currentPercent >= targetPercent) {
      currentPercent = targetPercent
      clearInterval(timer)
    }
    element.style.width = currentPercent + "%"
  }, 16)
}

// Lazy loading for images
function lazyLoadImages() {
  const images = document.querySelectorAll("img[data-src]")
  const imageObserver = new IntersectionObserver((entries, observer) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        const img = entry.target
        img.src = img.dataset.src
        img.classList.remove("lazy")
        imageObserver.unobserve(img)
      }
    })
  })

  images.forEach((img) => imageObserver.observe(img))
}

// Initialize lazy loading when DOM is ready
document.addEventListener("DOMContentLoaded", lazyLoadImages)

// Print functionality
function printReport() {
  window.print()
}

// Export data as JSON
function exportData(data, filename) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" })
  const url = URL.createObjectURL(blob)
  const a = document.createElement("a")
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}
