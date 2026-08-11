/* Live recommendation recompute.
   The pest identity is already settled; changing the field context only re-runs
   the rule engine. The server returns the rendered partial so this file never
   has to rebuild the recommendation markup. */

(function () {
  'use strict';

  const panel = document.getElementById('recomputePanel');
  const root = document.getElementById('recommendationRoot');
  if (!panel || !root) return;

  const status = document.getElementById('recomputeStatus');
  const notice = document.getElementById('recomputeNotice');

  const stage = document.getElementById('rc_stage');
  const severity = document.getElementById('rc_severity');
  const weather = document.getElementById('rc_weather');
  const days = document.getElementById('rc_days');
  const beneficials = document.getElementById('rc_beneficials');

  const endpoint = panel.dataset.endpoint;
  const aiClass = panel.dataset.aiClass;
  const confidence = parseFloat(panel.dataset.confidence);

  /* The context the page was rendered with. If the user returns to it we hide
     the "modified" notice again rather than leaving a stale message up. */
  const original = snapshot();
  let pending = null;
  let inFlight = false;

  function snapshot() {
    return {
      growth_stage: stage.value,
      severity: severity.value,
      weather: weather.value,
      days_to_harvest: days.value.trim() === '' ? null : days.value.trim(),
      beneficials_present: beneficials.checked
    };
  }

  function isOriginal(context) {
    return Object.keys(original).every(function (key) {
      return String(original[key]) === String(context[key]);
    });
  }

  function setBusy(busy) {
    status.classList.toggle('hidden', !busy);
    status.classList.toggle('inline-flex', busy);
    root.style.opacity = busy ? '0.55' : '1';
  }

  function showNotice(message, tone) {
    notice.textContent = message;
    notice.className =
      'mt-3 rounded-lg border px-3 py-2 text-xs ' +
      (tone === 'error'
        ? 'border-red-200 bg-red-50 text-red-800'
        : 'border-sky-200 bg-sky-50 text-sky-800');
  }

  function hideNotice() {
    notice.classList.add('hidden');
  }

  async function recompute() {
    if (inFlight) {
      pending = true;
      return;
    }
    inFlight = true;
    setBusy(true);

    const context = snapshot();

    try {
      const data = await window.CornGuard.postJSON(endpoint, {
        ai_class: aiClass,
        confidence: confidence,
        growth_stage: context.growth_stage,
        severity: context.severity,
        weather: context.weather,
        days_to_harvest: context.days_to_harvest,
        beneficials_present: context.beneficials_present
      });

      if (!data.ok) {
        showNotice(data.error || 'The rule engine could not be re-run.', 'error');
        return;
      }

      root.innerHTML = data.html;
      window.CornGuard.animateMeters(root);

      if (isOriginal(context)) {
        hideNotice();
      } else {
        showNotice(
          'Showing a what-if recommendation for a changed field context. The saved ' +
            'record still holds the original context, and the pest identity is unchanged.',
          'info'
        );
      }
    } catch (error) {
      showNotice('Could not reach the server. Check the connection and try again.', 'error');
    } finally {
      inFlight = false;
      setBusy(false);
      if (pending) {
        pending = false;
        recompute();
      }
    }
  }

  /* Selects and the checkbox fire immediately; the number input is debounced so
     typing "120" does not trigger three requests. */
  [stage, severity, weather, beneficials].forEach(function (control) {
    control.addEventListener('change', recompute);
  });

  let timer = null;
  days.addEventListener('input', function () {
    clearTimeout(timer);
    timer = setTimeout(recompute, 450);
  });
})();
