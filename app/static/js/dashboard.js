/* Dashboard behaviour: sortable per-class table, inline SVG training curves and
   a source-reload button. Charts are drawn by hand rather than pulled from a
   charting library so the page keeps its single-CDN footprint. */

(function () {
  'use strict';

  /* ------------------------------------------------------------------ *
   * Sortable per-class table
   * ------------------------------------------------------------------ */
  const table = document.getElementById('perClassTable');
  if (table) {
    const body = table.tBodies[0];
    let lastColumn = -1;
    let ascending = false;

    table.querySelectorAll('th[data-sort]').forEach(function (header) {
      header.addEventListener('click', function () {
        const column = parseInt(header.dataset.sort, 10);
        const numeric = header.dataset.type === 'number';
        ascending = column === lastColumn ? !ascending : numeric ? true : true;
        lastColumn = column;

        const rows = Array.prototype.slice.call(body.rows);
        rows.sort(function (a, b) {
          const cellA = a.cells[column];
          const cellB = b.cells[column];
          if (numeric) {
            const x = parseFloat(cellA.dataset.value || '0');
            const y = parseFloat(cellB.dataset.value || '0');
            return ascending ? x - y : y - x;
          }
          const textA = cellA.textContent.trim().toLowerCase();
          const textB = cellB.textContent.trim().toLowerCase();
          return ascending ? textA.localeCompare(textB) : textB.localeCompare(textA);
        });
        rows.forEach(function (row) { body.appendChild(row); });

        table.querySelectorAll('th[data-sort]').forEach(function (other) {
          other.classList.toggle('text-soil-800', other === header);
        });
      });
    });
  }

  /* ------------------------------------------------------------------ *
   * Training curves
   * ------------------------------------------------------------------ */
  const dataTag = document.getElementById('historyData');
  if (dataTag) {
    let history = [];
    try {
      history = JSON.parse(dataTag.textContent);
    } catch (error) {
      history = [];
    }

    if (history.length > 1) {
      drawChart(document.getElementById('accuracyChart'), history, [
        { key: 'accuracy', label: 'Training', colour: '#4d7c2f' },
        { key: 'val_accuracy', label: 'Validation', colour: '#0284c7' }
      ], { percent: true });

      drawChart(document.getElementById('lossChart'), history, [
        { key: 'loss', label: 'Training', colour: '#b45309' },
        { key: 'val_loss', label: 'Validation', colour: '#7c3aed' }
      ], { percent: false });
    }
  }

  /**
   * Render a small multi-series line chart into `host` as inline SVG.
   * The viewBox does the scaling, so the chart stays sharp at any width.
   */
  function drawChart(host, rows, series, options) {
    if (!host) return;

    const W = 520;
    const H = 240;
    const PAD = { top: 14, right: 12, bottom: 30, left: 44 };
    const plotW = W - PAD.left - PAD.right;
    const plotH = H - PAD.top - PAD.bottom;

    const present = series.filter(function (s) {
      return rows.some(function (row) { return typeof row[s.key] === 'number'; });
    });
    if (!present.length) return;

    let min = Infinity;
    let max = -Infinity;
    rows.forEach(function (row) {
      present.forEach(function (s) {
        const value = row[s.key];
        if (typeof value !== 'number') return;
        if (value < min) min = value;
        if (value > max) max = value;
      });
    });
    if (min === max) { min -= 0.05; max += 0.05; }
    const headroom = (max - min) * 0.1;
    min = Math.max(0, min - headroom);
    max = max + headroom;

    const epochs = rows.map(function (row, index) {
      return typeof row.epoch === 'number' ? row.epoch : index;
    });
    const firstEpoch = epochs[0];
    const lastEpoch = epochs[epochs.length - 1];
    const span = lastEpoch - firstEpoch || 1;

    function x(index) { return PAD.left + ((epochs[index] - firstEpoch) / span) * plotW; }
    function y(value) { return PAD.top + plotH - ((value - min) / (max - min)) * plotH; }

    function fmt(value) {
      return options.percent ? (value * 100).toFixed(0) + '%' : value.toFixed(2);
    }

    const parts = [];
    parts.push('<svg viewBox="0 0 ' + W + ' ' + H + '" class="w-full" role="img" aria-label="Training curve">');

    /* Gridlines and y-axis labels */
    for (let i = 0; i <= 4; i++) {
      const value = min + ((max - min) * i) / 4;
      const yy = y(value);
      parts.push('<line x1="' + PAD.left + '" y1="' + yy + '" x2="' + (W - PAD.right) + '" y2="' + yy +
                 '" stroke="#e7e3dc" stroke-width="1"/>');
      parts.push('<text x="' + (PAD.left - 8) + '" y="' + (yy + 3.5) +
                 '" text-anchor="end" font-size="9" fill="#9c9488">' + fmt(value) + '</text>');
    }

    /* X-axis labels at the ends and midpoint */
    [0, Math.floor((rows.length - 1) / 2), rows.length - 1].forEach(function (index) {
      parts.push('<text x="' + x(index) + '" y="' + (H - 10) +
                 '" text-anchor="middle" font-size="9" fill="#9c9488">' + epochs[index] + '</text>');
    });
    parts.push('<text x="' + (PAD.left + plotW / 2) + '" y="' + (H - 0.5) +
               '" text-anchor="middle" font-size="8" fill="#c4bdb2">epoch</text>');

    /* One polyline plus end-point dot per series */
    present.forEach(function (s) {
      const points = [];
      rows.forEach(function (row, index) {
        if (typeof row[s.key] === 'number') {
          points.push(x(index).toFixed(1) + ',' + y(row[s.key]).toFixed(1));
        }
      });
      if (!points.length) return;
      parts.push('<polyline fill="none" stroke="' + s.colour + '" stroke-width="2" ' +
                 'stroke-linejoin="round" stroke-linecap="round" points="' + points.join(' ') + '"/>');
      const last = points[points.length - 1].split(',');
      parts.push('<circle cx="' + last[0] + '" cy="' + last[1] + '" r="3" fill="' + s.colour + '"/>');
    });

    parts.push('</svg>');

    const legend = present.map(function (s) {
      const finalRow = rows[rows.length - 1];
      const finalValue = typeof finalRow[s.key] === 'number' ? fmt(finalRow[s.key]) : '—';
      return '<span class="flex items-center gap-1.5">' +
             '<span class="h-2 w-4 rounded-full" style="background:' + s.colour + '"></span>' +
             s.label + ' <span class="font-mono font-semibold text-soil-700">' + finalValue + '</span></span>';
    }).join('');

    host.innerHTML = parts.join('') +
      '<div class="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-soil-500">' + legend +
      '<span class="ml-auto text-soil-400">final epoch</span></div>';
  }

  /* ------------------------------------------------------------------ *
   * Reload the workbook and metrics without restarting the server
   * ------------------------------------------------------------------ */
  const reloadButton = document.getElementById('reloadSources');
  if (reloadButton) {
    const output = document.getElementById('reloadResult');
    reloadButton.addEventListener('click', async function () {
      reloadButton.disabled = true;
      reloadButton.classList.add('opacity-60');
      output.textContent = 'Reloading…';
      output.className = 'mt-2 text-center text-[11px] text-soil-500';

      try {
        const data = await window.CornGuard.postJSON('/api/reload', {});
        if (data.ok) {
          const kb = data.knowledge_base || {};
          output.textContent = 'Reloaded ' + (kb.pest_count || 0) + ' pests and ' +
                               (kb.product_count || 0) + ' products. Refresh to update the charts.';
          output.className = 'mt-2 text-center text-[11px] text-leaf-700';
        } else {
          output.textContent = data.error || 'Reload failed.';
          output.className = 'mt-2 text-center text-[11px] text-red-600';
        }
      } catch (error) {
        output.textContent = 'Could not reach the server.';
        output.className = 'mt-2 text-center text-[11px] text-red-600';
      } finally {
        reloadButton.disabled = false;
        reloadButton.classList.remove('opacity-60');
      }
    });
  }
})();
