const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');

const project = path.resolve(__dirname, '..');
const web = fs.readFileSync(path.join(project, 'app/ui/web.py'), 'utf8');
const code = web.match(/RUN_CLOCK = """<script>([\s\S]*?)<\/script>"""/)[1];
const admin = fs.readFileSync(path.join(project, 'app/ui/admin_panel.html'), 'utf8');
const adminCode = admin.match(/<script nonce="\{\{nonce\}\}">([\s\S]*?)<\/script>/)[1];
new vm.Script(adminCode);

function runClock(sleepTicksInMono, sleepMs) {
  let wall = 1_000_000, mono = 0, nextId = 0;
  const queue = new Map(), docEvents = {}, winEvents = {};
  const node = {dataset: {taClock: 'run:one', taElapsed: '10'}, textContent: ''};
  const document = {
    hidden: false,
    querySelectorAll: () => [node],
    addEventListener: (name, callback) => {docEvents[name] = callback;},
  };
  const window = {addEventListener: (name, callback) => {winEvents[name] = callback;}};
  const context = {document, window, performance: {now: () => mono}, Date: {now: () => wall},
    setTimeout(callback, delay) {const id = ++nextId; queue.set(id, {at: mono + delay, callback}); return id;},
    clearTimeout(id) {queue.delete(id);}};
  vm.runInNewContext(code, context);
  const advance = (milliseconds) => {
    const end = mono + milliseconds;
    while (true) {
      const next = [...queue.entries()].sort((a, b) => a[1].at - b[1].at)[0];
      if (!next || next[1].at > end) break;
      wall += next[1].at - mono;
      mono = next[1].at;
      queue.delete(next[0]);
      next[1].callback();
    }
    wall += end - mono;
    mono = end;
  };
  const fireOverdue = () => {
    const next = [...queue.entries()].sort((a, b) => a[1].at - b[1].at)[0];
    assert.ok(next && next[1].at <= mono);
    queue.delete(next[0]);
    next[1].callback();
  };
  assert.equal(node.textContent, '00:10');
  advance(1000);
  assert.equal(node.textContent, '00:11');
  advance(300);
  node.dataset.taElapsed = '11';
  advance(700);
  assert.equal(node.textContent, '00:12', 'poll must preserve the fractional second');
  node.dataset.taElapsed = '90';
  advance(200);
  assert.equal(node.textContent, '01:30', 'large server correction must apply');
  node.dataset.taElapsed = '3';
  advance(200);
  assert.equal(node.textContent, '01:30', 'stale server seconds must not rewind a running clock');
  document.hidden = true;
  docEvents.visibilitychange();
  wall += sleepMs;
  mono += sleepTicksInMono ? sleepMs : 100;
  document.hidden = false;
  docEvents.visibilitychange();
  assert.equal(node.textContent, sleepMs === 10_000 ? '01:40' : '01:33',
    'resume must include system sleep');
  node.dataset.taClock = 'run:two';
  node.dataset.taElapsed = '1';
  winEvents.pageshow();
  assert.equal(node.textContent, '00:01', 'new analysis must start a fresh timer');
  wall += 390;
  mono += 390;
  fireOverdue();
  assert.equal(node.textContent, '00:01', 'late callback must use elapsed time');
  advance(810);
  assert.equal(node.textContent, '00:02', 'late callback must not shift the next second by a full interval');
}
for (const sleepMs of [2800, 10_000]) {
  runClock(false, sleepMs);
  runClock(true, sleepMs);
}

let wall = 1_000_000, mono = 0;
const clockCode = admin.slice(admin.indexOf('const timebase ='), admin.indexOf('function ago('));
const context = vm.createContext({Date: {now: () => wall, parse: Date.parse}, performance: {now: () => mono}});
vm.runInContext(clockCode, context);
assert.equal(vm.runInContext('clockNow()', context), 1_000_000);
wall += 1000; mono += 1000;
assert.equal(vm.runInContext('clockNow()', context), 1_001_000);
wall += 2800; mono += 100;
assert.equal(vm.runInContext('clockNow()', context), 1_003_800,
  'short sleep must be counted on macOS and Linux');
wall += 10_000; mono += 100;
assert.equal(vm.runInContext('clockNow()', context), 1_013_800);
vm.runInContext('syncServerClock("1970-01-01T00:17:13.800Z", 100)', context);
assert.equal(vm.runInContext('clockNow()', context), 1_033_800, 'server time must correct client clock skew');
wall += 6_000; mono += 6_000;
const beforeSlowPoll = vm.runInContext('clockNow()', context);
vm.runInContext('syncServerClock("1970-01-01T00:17:14.300Z", 6000)', context);
assert.equal(vm.runInContext('clockNow()', context), beforeSlowPoll,
  'a delayed response with an old timestamp must not rewind the clock');
vm.runInContext('syncServerClock("1970-01-01T00:17:29.800Z", 100)', context);
assert.equal(vm.runInContext('clockNow()', context), 1_049_800,
  'a real server clock correction must still be applied');
console.log('Timer simulation passed: delayed callbacks, refresh, corrections, sleep, new run.');
