/* Shared front-end behaviour.
   Page-specific logic lives in its own file (upload.js, result.js, dashboard.js). */

(function () {
  'use strict';

  /* --- Mobile navigation ------------------------------------------------ */
  const toggle = document.getElementById('navToggle');
  const mobileNav = document.getElementById('mobileNav');
  if (toggle && mobileNav) {
    toggle.addEventListener('click', function () {
      const open = mobileNav.classList.toggle('hidden') === false;
      toggle.setAttribute('aria-expanded', String(open));
    });
  }

  /* --- Auto-dismiss success flashes ------------------------------------- */
  document.querySelectorAll('[role="status"] .border-leaf-200').forEach(function (el) {
    setTimeout(function () {
      el.style.transition = 'opacity .4s';
      el.style.opacity = '0';
      setTimeout(function () { el.remove(); }, 400);
    }, 6000);
  });
})();

/* --------------------------------------------------------------------------
 * Helpers shared by the page scripts.
 * ------------------------------------------------------------------------ */
window.CornGuard = {
  /** Format a 0-1 probability as a percentage string. */
  pct: function (value, digits) {
    if (value === null || value === undefined) return 'n/a';
    return (value * 100).toFixed(digits === undefined ? 1 : digits) + '%';
  },

  /** Tailwind classes for an action level. */
  actionStyle: function (level) {
    switch (level) {
      case 'treat_now':  return { bg: 'bg-red-50',     border: 'border-red-200',     text: 'text-red-800',     bar: 'bg-red-500' };
      case 'treat_soon': return { bg: 'bg-amber-50',   border: 'border-amber-200',   text: 'text-amber-800',   bar: 'bg-amber-500' };
      case 'preventive': return { bg: 'bg-sky-50',     border: 'border-sky-200',     text: 'text-sky-800',     bar: 'bg-sky-500' };
      default:           return { bg: 'bg-leaf-50',    border: 'border-leaf-200',    text: 'text-leaf-800',    bar: 'bg-leaf-500' };
    }
  },

  /** Escape text before inserting it into innerHTML. */
  esc: function (value) {
    const div = document.createElement('div');
    div.textContent = value === null || value === undefined ? '' : String(value);
    return div.innerHTML;
  },

  /** Animate a meter's width once it scrolls into view. */
  animateMeters: function (root) {
    (root || document).querySelectorAll('[data-meter]').forEach(function (el) {
      const target = el.getAttribute('data-meter');
      requestAnimationFrame(function () {
        requestAnimationFrame(function () { el.style.width = target + '%'; });
      });
    });
  },

  /** POST JSON and return the parsed body. */
  postJSON: async function (url, payload) {
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    return response.json();
  }
};

document.addEventListener('DOMContentLoaded', function () {
  window.CornGuard.animateMeters(document);
});
