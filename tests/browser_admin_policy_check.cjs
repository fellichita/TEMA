const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, '../app/ui/admin_panel.html'), 'utf8');
const start = html.indexOf('function olderRadarPolicy(');
const end = html.indexOf('function renderVisitors()', start);
assert.ok(start >= 0 && end > start, 'run drawer script must be present');
const code = html.slice(start, end);
const notice = 'Этот сохранённый анализ создан до обновления отбора направлений. Запустите новый анализ, чтобы увидеть результаты по новым правилам.';

function render(version) {
  let output = '';
  const context = {
    runData: {
      state: 'succeeded', query: 'Battery research', id: 'run-one',
      radar_policy_version: version, numbers: {technologies: 1, technology_candidates: 1},
      technologies: [{title: 'Legacy mixed category', probability: 0.85}],
    },
    $: () => ({}),
    setHTML: (_node, markup) => {output = markup;},
    esc: String,
    badge: () => 'Готово',
    when: () => '',
    duration: () => '',
    MODES: {},
    NF: new Intl.NumberFormat('ru-RU'),
    pubs: {},
  };
  vm.runInNewContext(code, context);
  vm.runInNewContext('renderRun()', context);
  return output;
}

for (const version of ['radar/1.1.0', 'radar/1.2.0']) {
  const markup = render(version);
  assert.ok(markup.includes(notice), `old ${version} run needs policy notice`);
  assert.ok(markup.includes('Legacy mixed category'), 'saved technology should still render');
}
for (const version of ['radar/1.3.0', 'radar/2.0.0', 'radar/1.2.0<script>', null]) {
  assert.ok(!render(version).includes(notice), `${version} must not get the old-policy notice`);
}
