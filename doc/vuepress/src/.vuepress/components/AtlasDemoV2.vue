<template>
  <div class="atlasd">
    <p class="atlasd-note">
      A simulation that runs in your browser. No real agent or GPU is running, and all numbers and file contents are samples.
    </p>

    <ol class="atlasd-steps">
      <li v-for="(s, i) in steps" :key="i" :class="{ done: stepDone[i], now: stepNow === i }">
        <span class="atlasd-step-n">{{ i + 1 }}</span>
        <span>{{ s }}</span>
      </li>
    </ol>

    <div class="atlasd-cmds">
      <button class="atlasd-btn" :class="{ 'is-next': phase === 'new' }" :disabled="inited" @click="runInit">atlas init</button>
      <button class="atlasd-btn" :class="{ 'is-next': phase === 'ready' }" :disabled="phase !== 'ready'" :title="inited ? '' : 'Run init first'" @click="runRun">atlas run</button>
      <button class="atlasd-btn" :disabled="!inited" :title="inited ? '' : 'Run init first'" @click="runStatus">atlas status</button>
      <button class="atlasd-btn atlasd-btn-stop" :disabled="!canStop" @click="runStop">atlas stop</button>
      <span class="atlasd-spacer"></span>
      <span class="atlasd-speed" role="group" aria-label="Speed">
        <span class="atlasd-speed-label">Speed</span>
        <button v-for="s in [1, 2]" :key="s" class="atlasd-chip" :class="{ on: speed === s }" @click="speed = s">{{ s }}x</button>
      </span>
      <button class="atlasd-btn atlasd-btn-quiet" @click="reset">Reset</button>
    </div>

    <div class="atlasd-row">
      <section class="atlasd-term" ref="term" role="log" aria-live="off">
        <div v-if="!log.length" class="atlasd-line k-dim">Start with atlas init</div>
        <div v-for="l in log" :key="l.id" class="atlasd-line" :class="'k-' + l.k">{{ l.k === 'cmd' ? '$ ' : '' }}{{ l.t }}</div>
      </section>

      <section class="atlasd-card atlasd-viewer" ref="viewerCard">
        <header class="atlasd-head">
          <span class="atlasd-fname">{{ sel || 'No file open' }}</span>
          <span class="atlasd-head-hint">{{ viewHint }}</span>
        </header>
        <textarea
          v-if="editable"
          v-model="program"
          class="atlasd-editor"
          spellcheck="false"
          aria-label="program.md"
        ></textarea>
        <pre v-else-if="sel" ref="viewer" class="atlasd-editor">{{ viewText }}</pre>
        <div v-else class="atlasd-placeholder"><p>No files yet. Run <code>atlas init</code> to create the study directory.</p></div>
      </section>
    </div>

    <div class="atlasd-row">
      <section class="atlasd-card atlasd-treecard">
        <header class="atlasd-head">
          <strong>Study directory</strong>
          <span class="atlasd-head-hint">click a file to open it</span>
        </header>
        <transition-group tag="ul" name="atlasd-tree" class="atlasd-tree">
          <li
            v-for="e in tree"
            :key="e.id"
            :class="[e.k, { special: e.sp, sel: e.path && e.path === sel, link: !!e.path }]"
            :style="{ paddingLeft: 12 + e.d * 14 + 'px' }"
            :tabindex="e.path ? 0 : -1"
            :role="e.path ? 'button' : null"
            @click="open(e)"
            @keyup.enter="open(e)"
          >{{ e.p }}</li>
          <li v-if="!tree.length" key="empty" class="atlasd-empty">Created by init</li>
        </transition-group>
      </section>

      <section class="atlasd-card">
        <header class="atlasd-head">
          <strong>Workflow graph</strong>
          <span class="atlasd-legend">
            <i class="sw sw-agent"></i>calls the agent
            <i class="sw sw-local"></i>local step
          </span>
        </header>
        <div class="atlasd-tickrow" :class="{ off: !isRunning }">
          <span>controller tick</span>
          <span class="atlasd-tickbar"><b :style="{ width: tickPct + '%' }"></b></span>
          <span class="atlasd-tickhint">dashed ring = waiting for the next tick</span>
        </div>
        <svg class="atlasd-graph" viewBox="0 0 640 320" role="img" aria-label="Workflow graph">
          <defs>
            <marker id="atlasd-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
              <path d="M1 1 L9 5 L1 9 z" class="atlasd-arrowhead" />
            </marker>
          </defs>
          <g>
            <path v-for="(e, i) in edges" :key="'e' + i" :d="e.d" class="atlasd-edge" :class="{ dashed: e.dashed }" marker-end="url(#atlasd-arrow)" />
            <text v-for="(e, i) in edgeLabels" :key="'l' + i" :x="e.lx" :y="e.ly" class="atlasd-edge-label">{{ e.label }}</text>
          </g>
          <g v-for="(n, key) in nodes" :key="key" class="atlasd-node" :class="nodeClass(key, n)">
            <rect :x="n.x - 60" :y="n.y - 17" width="120" height="34" rx="6" />
            <text :x="n.x" :y="n.y + 4" text-anchor="middle">{{ key }}</text>
          </g>
          <g v-for="tr in activeTrials" :key="'t' + tr.n" class="atlasd-token" :style="tokenStyle(tr)">
            <circle v-if="tr.ready" r="14.5" class="atlasd-wait" />
            <circle r="10" :fill="hue(tr.n)" />
            <text y="4" text-anchor="middle">{{ tr.n }}</text>
          </g>
        </svg>
      </section>
    </div>

    <section class="atlasd-card atlasd-gpucard">
      <header class="atlasd-head"><strong>GPU jobs</strong></header>
      <div class="atlasd-gpus">
        <div v-for="(g, i) in gpus" :key="i" class="atlasd-gpu" :class="'s-' + g.state">
          <div class="atlasd-gpu-top">
            <span class="atlasd-gpu-name">GPU {{ i }}</span>
            <span class="atlasd-gpu-state">{{ g.state }}</span>
          </div>
          <div class="atlasd-gpu-trial">
            <template v-if="g.trial">trial {{ g.trial }} · epoch {{ trials[g.trial - 1].epoch }}/{{ epochs }}</template>
            <template v-else>&nbsp;</template>
          </div>
          <div class="atlasd-meter"><span class="lbl">util</span><span class="bar"><b :style="{ width: g.util + '%' }"></b></span><span class="val">{{ Math.round(g.util) }}%</span></div>
          <div class="atlasd-meter"><span class="lbl">mem</span><span class="bar"><b :style="{ width: g.mem + '%' }"></b></span><span class="val">{{ Math.round(g.mem) }}%</span></div>
          <svg class="atlasd-spark" viewBox="0 0 120 36" preserveAspectRatio="none" aria-hidden="true">
            <polyline v-if="g.trial" :points="spark(trials[g.trial - 1])" />
          </svg>
        </div>
      </div>
    </section>

    <section class="atlasd-card atlasd-trials">
      <div class="atlasd-trials-chart">
        <header class="atlasd-head">
          <strong>Trials</strong>
          <span class="atlasd-head-hint">best dev_wer: {{ best.toFixed(2) }} (sample)</span>
        </header>
        <svg class="atlasd-chart" viewBox="0 0 260 66" role="img" aria-label="Best dev_wer so far">
          <line x1="8" x2="252" :y1="chartY(baseline)" :y2="chartY(baseline)" class="atlasd-base" />
          <text x="252" :y="chartY(baseline) - 3" text-anchor="end" class="atlasd-base-label">baseline {{ baseline.toFixed(2) }}</text>
          <polyline :points="bestLine" class="atlasd-bestline" />
          <circle v-for="tr in scored" :key="'p' + tr.n" :cx="chartX(tr.n)" :cy="chartY(tr.score)" r="4" :class="tr.final === 'accepted' ? 'pt-ok' : 'pt-no'" />
        </svg>
      </div>
      <table class="atlasd-table">
        <tbody>
          <tr v-for="tr in trials" :key="tr.n">
            <td class="c-n"><i class="dot" :style="{ background: hue(tr.n) }"></i>{{ tr.n }}</td>
            <td class="c-patch">{{ tr.patchText || '…' }}</td>
            <td class="c-status"><span class="atlasd-pill" :class="'p-' + (tr.final || 'running')">{{ statusText(tr) }}</span></td>
            <td class="c-score">{{ tr.score === null ? '—' : tr.score.toFixed(2) }}</td>
          </tr>
          <tr v-if="!trials.length"><td colspan="4" class="atlasd-empty">Trials appear after run</td></tr>
        </tbody>
      </table>
    </section>
  </div>
</template>

<script>
const EPOCHS = 8
const BASELINE = 8.4
const STUDY = 'atlas_study'
const TICK_EVERY = 1.5
const DUR = { pending: 1.2, train: 8 }
// How long a node's own work takes. When it is done, the node waits for the next controller tick.
const WORK = { propose: 1.0, debug: 1.0, collect: 0, decide: 0, reflect: 0.8, knowledge: 0.6, budget: 0 }
const PHASE_WORK = { context: 1.0, select: 0, analyze: 1.2 }
const STAGE_NODE = {
  propose: 'propose_trial',
  pending: 'run_trial',
  train: 'run_trial',
  debug: 'debug_trial',
  collect: 'collect_result',
  decide: 'accept_or_reject',
  reflect: 'reflect',
  knowledge: 'update_knowledge',
  budget: 'check_budget'
}
const STAGE_LABEL = {
  propose: 'proposing', pending: 'job pending', train: 'training', debug: 'debugging',
  collect: 'collecting', decide: 'deciding', reflect: 'reflecting', knowledge: 'updating notes', budget: 'checking budget'
}
const FINAL_LABEL = { accepted: 'accepted', rejected: 'rejected', early_stopped: 'early stopped' }
const NODES = {
  research_context: { x: 80, y: 100, agent: true },
  propose_trial: { x: 230, y: 100, agent: true },
  run_trial: { x: 380, y: 100 },
  collect_result: { x: 530, y: 100 },
  accept_or_reject: { x: 530, y: 190 },
  reflect: { x: 380, y: 190, agent: true },
  update_knowledge: { x: 230, y: 190, agent: true },
  check_budget: { x: 80, y: 190 },
  select_best: { x: 80, y: 280 },
  analyze: { x: 230, y: 280, agent: true },
  debug_trial: { x: 380, y: 34, agent: true }
}
const EDGES = [
  { d: 'M140 100 L170 100' },
  { d: 'M290 100 L320 100' },
  { d: 'M440 100 L470 100' },
  { d: 'M530 117 L530 173' },
  { d: 'M470 190 L440 190' },
  { d: 'M320 190 L290 190' },
  { d: 'M170 190 L140 190' },
  { d: 'M80 207 L80 263' },
  { d: 'M140 280 L170 280' },
  { d: 'M104 173 C 124 148, 188 148, 206 117', dashed: true, label: 'budget left', lx: 112, ly: 150 },
  { d: 'M366 83 L366 55', label: 'failed', lx: 330, ly: 72 },
  { d: 'M394 55 L394 83', label: 'fixed', lx: 402, ly: 72 }
]
const FLOW = [
  ['research_context', 'propose_trial'], ['propose_trial', 'run_trial'], ['run_trial', 'collect_result'],
  ['run_trial', 'debug_trial'], ['debug_trial', 'run_trial'], ['collect_result', 'accept_or_reject'],
  ['accept_or_reject', 'reflect'], ['reflect', 'update_knowledge'], ['update_knowledge', 'check_budget'],
  ['check_budget', 'propose_trial'], ['check_budget', 'select_best'], ['select_best', 'analyze']
]
const HUES = ['#3B7DDD', '#C2417F', '#2E9C8F', '#C98500', '#7A5AD1', '#D1583A']

const CANDIDATES = [
  { id: 'optimizer', kw: ['optimizer', 'adamw', 'adam'], pairs: [['optimizer.type', 'adamw']], delta: -0.24 },
  { id: 'specaug', kw: ['specaug', 'augment', 'mask'], pairs: [['frontend.specaug.time_mask_width_ratio_range', '[0, 0.08]']], delta: -0.14 },
  { id: 'dropout', kw: ['dropout', 'regulariz'], pairs: [['model.encoder.dropout_rate', '0.15']], delta: -0.05 },
  { id: 'scheduler', kw: ['scheduler', 'warmup', 'learning rate'], pairs: [['scheduler.warmup_steps', '25000']], delta: -0.02 },
  { id: 'batch', kw: ['batch'], pairs: [['dataloader.batch_bins', '16000000']], delta: 0.12 },
  { id: 'ctc', kw: ['ctc'], pairs: [['model.ctc_weight', '0.3']], delta: -0.3 },
  { id: 'decay', kw: ['weight decay', 'decay'], pairs: [['optimizer.weight_decay', '0.01']], delta: 0.03 },
  { id: 'clip', kw: ['clip', 'gradient'], pairs: [['trainer.gradient_clip_val', '5.0']], delta: -0.04 }
]
const ALLOWED = ['optimizer.', 'scheduler.', 'model.', 'dataloader.', 'frontend.', 'trainer.']
const DENIED = ['dataset.']

const DEFAULT_PROGRAM = `# Objective
Lower dev_wer of the LibriSpeech-100 recipe.

# Already tuned
- learning rate and warmup

# Worth trying
- optimizer choice
- SpecAugment masks
- dropout

# Off limits
- anything under dataset.*
`

const YAML = `atlas:
  application: hyperparameter_search
  agent:
    provider: claude
  submission:
    controller:
      run_in: local
  budget:
    max_trials: 6
    max_parallel_trials: 2
  search_space:
    allowed_keys: [optimizer.*, scheduler.*, model.*, dataloader.*, trainer.*]
    denied_keys: [dataset.*]
`

function graphYaml() {
  let s = '# Simplified for this demo. The real graph.yaml also defines each stage.\nnodes:\n'
  Object.keys(NODES).forEach(function (k) { s += '  - ' + k + (NODES[k].agent ? '   # agent\n' : '\n') })
  s += 'edges:\n'
  FLOW.forEach(function (e) { s += '  - ' + e[0] + ' -> ' + e[1] + '\n' })
  return s
}

function mulberry32(a) {
  return function () {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function pad(n, w) {
  let s = String(n)
  while (s.length < w) s = '0' + s
  return s
}

export default {
  name: 'AtlasDemoV2',
  props: {
    maxTrials: { type: Number, default: 6 },
    maxParallel: { type: Number, default: 2 }
  },
  data() {
    return {
      nodes: NODES,
      edges: EDGES,
      baseline: BASELINE,
      epochs: EPOCHS,
      gpus: this.makeGpus(),
      steps: ['init', 'Edit program.md', 'run', 'status / stop'],
      program: '',
      savedProgram: '',
      sel: '',
      phase: 'new', // new, ready, context, search, select, analyze, done, stopped
      phaseT: 0,
      inited: false,
      speed: 1,
      stopRequested: false,
      stopFile: false,
      knowledgePack: false,
      summary: false,
      best: BASELINE,
      trials: [],
      log: [],
      logSeq: 0,
      jobSeq: 0,
      tickAcc: 0,
      timer: null,
      last: 0
    }
  },
  computed: {
    edgeLabels() {
      return EDGES.filter(function (e) { return e.label })
    },
    tickPct() {
      return this.isRunning ? Math.min(100, (this.tickAcc / TICK_EVERY) * 100) : 0
    },
    isRunning() {
      return ['context', 'search', 'select', 'analyze'].indexOf(this.phase) >= 0
    },
    canStop() {
      return this.inited && !this.stopRequested && (this.phase === 'context' || this.phase === 'search')
    },
    stepDone() {
      return [this.inited, this.phase !== 'new' && this.phase !== 'ready', this.phase === 'done' || this.phase === 'stopped', false]
    },
    stepNow() {
      if (this.phase === 'new') return 0
      if (this.phase === 'ready') return 1
      if (this.isRunning) return 3
      return -1
    },
    activeTrials() {
      return this.trials.filter(function (x) { return !x.done })
    },
    scored() {
      return this.trials.filter(function (x) { return x.score !== null && x.final })
    },
    bestLine() {
      let b = BASELINE
      const pts = [this.chartX(0) + ',' + this.chartY(b)]
      const self = this
      this.trials.forEach(function (tr) {
        if (tr.final === 'accepted') b = tr.score
        if (tr.final) pts.push(self.chartX(tr.n) + ',' + self.chartY(b))
      })
      return pts.join(' ')
    },
    slots() {
      const seen = {}
      const out = {}
      this.activeTrials.forEach(function (tr) {
        seen[tr.node] = seen[tr.node] || 0
        out[tr.n] = seen[tr.node]
        seen[tr.node] += 1
      })
      return out
    },
    activeNodes() {
      const s = {}
      this.activeTrials.forEach(function (tr) { s[tr.node] = tr.stage })
      if (this.phase === 'context') s.research_context = 'agent'
      if (this.phase === 'select') s.select_best = 'local'
      if (this.phase === 'analyze') s.analyze = 'agent'
      return s
    },
    editable() {
      return this.phase === 'ready' && this.sel === 'program.md'
    },
    liveTrial() {
      const m = /^trials\/trial_(\d+)\/train\.log$/.exec(this.sel)
      if (!m) return null
      const tr = this.trials[parseInt(m[1], 10) - 1]
      return tr && (tr.stage === 'pending' || tr.stage === 'train') && !tr.done ? tr : null
    },
    viewHint() {
      if (!this.sel) return ''
      if (this.editable) return 'created by init: edit it, then run'
      if (this.liveTrial) return 'live: grows while the job runs'
      return 'read-only'
    },
    viewText() {
      const t = this.fileContent(this.sel)
      return t === '' ? '(empty file)' : t
    },
    tree() {
      if (!this.inited) return []
      const out = []
      const add = function (id, k, p, d, path, sp) { out.push({ id: id, k: k, p: p, d: d, path: path || '', sp: !!sp }) }
      add('root', 'dir', STUDY + '/', 0)
      add('atlas', 'file', 'atlas.yaml', 1, 'atlas.yaml')
      add('graph', 'file', 'graph.yaml', 1, 'graph.yaml')
      add('prog', 'file', 'program.md', 1, 'program.md')
      if (this.knowledgePack) add('kp', 'file', 'knowledge_pack.md', 1, 'knowledge_pack.md')
      if (this.trials.length) {
        add('csv', 'file', 'trials.csv', 1, 'trials.csv')
        add('tdir', 'dir', 'trials/', 1)
      }
      this.trials.forEach(function (tr) {
        if (!tr.patchReady) return
        const dir = 'trials/trial_' + pad(tr.n, 6) + '/'
        const f = function (id, name, sp) { add('t' + tr.n + id, 'file', name, 3, dir + name, sp) }
        add('t' + tr.n, 'dir', 'trial_' + pad(tr.n, 6) + '/', 2)
        f('p', 'config_patch.yaml')
        if (tr.job) f('l', 'train.log')
        if (tr.debugNote) f('g', 'debug.md')
        if (tr.earlyFile) f('e', 'EARLY_STOP_REQUESTED', true)
        if (tr.decided) f('d', 'decision.md')
        if (tr.reflected) f('r', 'reflection.md')
      })
      if (this.stopFile) add('stop', 'file', 'STOP', 1, 'STOP', true)
      if (this.summary) add('sum', 'file', 'summary.md', 1, 'summary.md')
      return out
    }
  },
  watch: {
    viewText() {
      if (!/\.log$/.test(this.sel)) return
      const self = this
      this.$nextTick(function () {
        const el = self.$refs.viewer
        if (el) el.scrollTop = el.scrollHeight
      })
    }
  },
  beforeDestroy() {
    this.stopTimer()
  },
  beforeUnmount() {
    this.stopTimer()
  },
  methods: {
    makeGpus() {
      const g = []
      for (let i = 0; i < this.maxParallel; i++) g.push({ state: 'idle', trial: 0, util: 0, mem: 0 })
      return g
    },
    acquireGpu(tr) {
      const i = this.gpus.findIndex(function (g) { return g.state === 'idle' })
      if (i >= 0) {
        this.gpus[i].state = 'pending'
        this.gpus[i].trial = tr.n
      }
    },
    releaseGpu(tr) {
      this.gpus.forEach(function (g) {
        if (g.trial === tr.n) {
          g.state = 'idle'
          g.trial = 0
        }
      })
    },
    updateGpus() {
      this.gpus.forEach(function (g) {
        const target = g.state === 'running' ? 86 + Math.random() * 12 : g.state === 'pending' ? 3 : 0
        const memT = g.state === 'running' ? 68 + Math.random() * 8 : g.state === 'pending' ? 6 : 0
        g.util += (target - g.util) * 0.4
        g.mem += (memT - g.mem) * 0.4
        if (g.util < 0.5) g.util = 0
        if (g.mem < 0.5) g.mem = 0
      })
    },
    spark(tr) {
      if (!tr || !tr.loss.length) return ''
      return tr.loss
        .map(function (v, i) {
          const x = (i * 120) / (EPOCHS - 1)
          const y = 34 - Math.min(1, Math.max(0, (v - 0.8) / 2.6)) * 32
          return x.toFixed(1) + ',' + y.toFixed(1)
        })
        .join(' ')
    },
    say(k, text) {
      this.logSeq += 1
      this.log.push({ id: this.logSeq, k: k, t: text })
      if (this.log.length > 160) this.log.splice(0, this.log.length - 160)
      const self = this
      this.$nextTick(function () {
        const el = self.$refs.term
        if (el) el.scrollTop = el.scrollHeight
      })
    },
    open(e) {
      if (!e.path) return
      this.sel = e.path
      const self = this
      this.$nextTick(function () {
        const el = self.$refs.viewerCard
        if (el && el.scrollIntoView) el.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
      })
    },
    hue(n) {
      return HUES[(n - 1) % HUES.length]
    },
    nodeClass(key, n) {
      const c = [n.agent ? 'is-agent' : 'is-local']
      if (this.activeNodes[key]) c.push('is-active')
      if (this.activeNodes[key] === 'train') c.push('is-gpu')
      return c
    },
    tokenStyle(tr) {
      const n = NODES[tr.node]
      const slot = this.slots[tr.n] || 0
      return { transform: 'translate(' + (n.x + 44 - slot * 24) + 'px,' + (n.y - 17) + 'px)' }
    },
    statusText(tr) {
      return tr.final ? FINAL_LABEL[tr.final] : STAGE_LABEL[tr.stage]
    },
    chartX(n) {
      return 16 + n * ((244 - 16) / Math.max(1, this.maxTrials))
    },
    chartY(v) {
      const lo = 7.7
      const hi = 8.7
      const c = Math.min(hi, Math.max(lo, v))
      return 60 - ((c - lo) / (hi - lo)) * 52
    },

    /* ---------- simulated file contents ---------- */
    fileContent(path) {
      switch (path) {
        case 'atlas.yaml': return YAML
        case 'graph.yaml': return graphYaml()
        case 'program.md': return this.phase === 'ready' ? this.program : this.savedProgram
        case 'STOP': return ''
        case 'knowledge_pack.md': return this.knowledgeText()
        case 'trials.csv': return this.csvText()
        case 'summary.md': return this.summaryText()
      }
      const m = /^trials\/trial_(\d+)\/(.+)$/.exec(path)
      if (!m) return ''
      const tr = this.trials[parseInt(m[1], 10) - 1]
      if (!tr) return ''
      switch (m[2]) {
        case 'config_patch.yaml': return this.patchText(tr)
        case 'train.log': return tr.logLines.join('\n')
        case 'debug.md': return tr.debugNote
        case 'EARLY_STOP_REQUESTED': return ''
        case 'decision.md': return tr.decisionText
        case 'reflection.md': return this.reflectionText(tr)
      }
      return ''
    },
    patchText(tr) {
      let s = 'config_patches:\n'
      tr.pairs.forEach(function (p) { s += '  - key: ' + p[0] + '\n    value: ' + p[1] + '\n' })
      if (tr.removed.length) s += '# removed by search_space (denied_keys): ' + tr.removed.join(', ') + '\n'
      return s
    },
    reflectionText(tr) {
      if (tr.final === 'accepted') {
        return '# Reflection on trial ' + tr.n + '\n\nImproved dev_wer to ' + tr.score.toFixed(2) + ' (best before: ' + tr.bestBefore.toFixed(2) + ').\nKeep: ' + tr.patchText + '\nNext: combine it with another untried item from program.md.\n'
      }
      if (tr.final === 'early_stopped') {
        return '# Reflection on trial ' + tr.n + '\n\nLoss rose from epoch 3 and the monitor stopped the job.\nNext: try a smaller change or a lower learning rate.\n'
      }
      return '# Reflection on trial ' + tr.n + '\n\nNo improvement (' + tr.score.toFixed(2) + ' vs best ' + tr.bestBefore.toFixed(2) + ').\n' + tr.patchText + ' did not help on its own. Lower its priority.\n'
    },
    knowledgeText() {
      const first = this.savedProgram.split('\n').filter(function (l) { return l.trim() && !/^\s*#/.test(l) })[0] || ''
      let s = '# Knowledge pack\n\n## Context\nObjective (from program.md): ' + first + '\nBaseline dev_wer: ' + BASELINE.toFixed(2) + ' (sample)\nAllowed: ' + ALLOWED.map(function (a) { return a + '*' }).join(', ') + '\nDenied: ' + DENIED.map(function (a) { return a + '*' }).join(', ') + '\n'
      const noted = this.trials.filter(function (tr) { return tr.noted })
      if (noted.length) s += '\n## Notes\n'
      noted.forEach(function (tr) {
        s += '- trial ' + tr.n + ' (' + FINAL_LABEL[tr.final] + (tr.score === null ? '' : ', ' + tr.score.toFixed(2)) + '): ' + tr.patchText + '\n'
      })
      return s
    },
    csvText() {
      let s = 'trial,status,dev_wer,config_patch\n'
      this.trials.forEach(function (tr) {
        s += tr.n + ',' + (tr.final || 'running') + ',' + (tr.score === null ? '' : tr.score.toFixed(2)) + ',"' + tr.patchText + '"\n'
      })
      return s
    },
    summaryText() {
      let bestTr = null
      this.trials.forEach(function (tr) { if (tr.final === 'accepted') bestTr = tr })
      let s = '# Summary\n\n' + (bestTr ? 'Best trial: ' + bestTr.n + ' (dev_wer ' + bestTr.score.toFixed(2) + ', baseline ' + BASELINE.toFixed(2) + ')\n' : 'No trial beat the baseline.\n') + '\n| trial | change | status | dev_wer |\n|---|---|---|---|\n'
      this.trials.forEach(function (tr) {
        s += '| ' + tr.n + ' | ' + tr.patchText + ' | ' + FINAL_LABEL[tr.final] + ' | ' + (tr.score === null ? '-' : tr.score.toFixed(2)) + ' |\n'
      })
      return s
    },

    /* ---------- commands ---------- */
    runInit() {
      if (this.inited) return
      this.rng = mulberry32(7)
      this.program = DEFAULT_PROGRAM
      this.savedProgram = DEFAULT_PROGRAM
      this.say('cmd', 'espnet3 atlas init --recipe-dir . --config conf/atlas.yaml')
      this.say('out', 'created ' + STUDY + '/ (atlas.yaml, graph.yaml, program.md)')
      this.inited = true
      this.phase = 'ready'
      this.sel = 'program.md'
    },
    runRun() {
      if (this.phase !== 'ready') return
      this.say('cmd', 'espnet3 atlas run --study-dir ' + STUDY + ' --recipe-dir .')
      this.savedProgram = this.program
      this.say('out', 'controller started (run_in=local)')
      this.phase = 'context'
      this.phaseT = 0
      this.tickAcc = TICK_EVERY - 0.3
      this.startTimer()
    },
    runStatus() {
      if (!this.inited) return
      const c = { accepted: 0, rejected: 0, early_stopped: 0, running: 0 }
      this.trials.forEach(function (tr) {
        if (tr.final) c[tr.final] += 1
        else c.running += 1
      })
      const label = { new: 'not initialized', ready: 'initialized', context: 'running', search: 'running', select: 'running', analyze: 'running', done: 'completed', stopped: 'stopped' }[this.phase]
      this.say('cmd', 'espnet3 atlas status --study-dir ' + STUDY)
      this.say('out', 'study: ' + STUDY + ' (' + label + ')')
      this.say('out', 'trials: accepted=' + c.accepted + ' rejected=' + c.rejected + ' early_stopped=' + c.early_stopped + ' running=' + c.running)
      this.say('out', 'STOP file: ' + (this.stopFile ? 'present' : 'absent'))
      const jobs = this.activeTrials.filter(function (tr) { return tr.stage === 'pending' || tr.stage === 'train' })
      if (!jobs.length) {
        this.say('out', 'active jobs: none')
      } else {
        this.say('out', 'active jobs:')
        const self = this
        jobs.forEach(function (tr) {
          self.say('out', '  ' + tr.job + '  trial ' + tr.n + '  ' + (tr.stage === 'train' ? 'RUNNING' : 'PENDING'))
        })
      }
    },
    runStop() {
      if (!this.canStop) return
      this.say('cmd', 'espnet3 atlas stop --study-dir ' + STUDY)
      this.say('out', 'STOP file written')
      this.stopFile = true
      this.stopRequested = true
    },
    reset() {
      this.stopTimer()
      this.phase = 'new'
      this.phaseT = 0
      this.inited = false
      this.stopRequested = false
      this.stopFile = false
      this.knowledgePack = false
      this.summary = false
      this.best = BASELINE
      this.trials = []
      this.gpus = this.makeGpus()
      this.log = []
      this.jobSeq = 0
      this.tickAcc = 0
      this.sel = ''
      this.program = ''
      this.savedProgram = ''
    },

    /* ---------- simulation loop ---------- */
    startTimer() {
      this.stopTimer()
      this.last = Date.now()
      this.timer = setInterval(this.onTimer, 100)
    },
    stopTimer() {
      if (this.timer) clearInterval(this.timer)
      this.timer = null
    },
    onTimer() {
      const now = Date.now()
      const dt = Math.min(0.3, (now - this.last) / 1000) * this.speed
      this.last = now
      this.tick(dt)
    },
    tick(dt) {
      this.updateGpus()
      if (this.phase === 'context' || this.phase === 'select' || this.phase === 'analyze') this.phaseT += dt
      if (this.phase === 'search') {
        const self = this
        this.activeTrials.forEach(function (tr) { self.work(tr, dt) })
      }
      if (!this.isRunning) return
      this.tickAcc += dt
      if (this.tickAcc >= TICK_EVERY) {
        this.tickAcc = 0
        this.controllerTick()
      }
    },
    phaseReady() {
      return this.phaseT >= (PHASE_WORK[this.phase] || 0)
    },
    controllerTick() {
      const waiting = this.activeTrials.filter(function (x) { return x.ready }).length
      const pend = waiting + (this.phase !== 'search' && this.phaseReady() ? 1 : 0)
      this.say('dim', '[tick] running_trials=' + this.activeTrials.length + ' pending_node_runs=' + pend)

      if (this.phase === 'context') {
        if (!this.phaseReady()) return
        this.knowledgePack = true
        this.phase = 'search'
      }
      if (this.phase === 'search') {
        const self = this
        this.activeTrials.filter(function (x) { return x.ready }).forEach(function (tr) { self.step(tr) })
        if (this.stopRequested) {
          if (!this.activeTrials.length) {
            this.say('out', 'STOP file found. Controller exits.')
            this.phase = 'stopped'
            this.stopTimer()
          }
          return
        }
        while (this.trials.length < this.maxTrials && this.activeTrials.length < this.maxParallel) this.createTrial()
        if (this.trials.length >= this.maxTrials && !this.activeTrials.length) {
          this.phase = 'select'
          this.phaseT = 0
        }
        return
      }
      if (this.phase === 'select') {
        if (!this.phaseReady()) return
        this.phase = 'analyze'
        this.phaseT = 0
        return
      }
      if (this.phase === 'analyze' && this.phaseReady()) {
        this.summary = true
        this.say('out', 'All trials finished. Controller exits.')
        this.phase = 'done'
        this.stopTimer()
      }
    },

    /* ---------- trials ---------- */
    rankCandidates() {
      const text = this.focusText()
      const scored = CANDIDATES.map(function (c, i) {
        let m = 0
        c.kw.forEach(function (k) { if (text.indexOf(k) >= 0) m += 1 })
        return { c: c, m: m, i: i }
      })
      scored.sort(function (a, b) { return b.m - a.m || a.i - b.i })
      return scored.map(function (x) { return x.c })
    },
    focusText() {
      const lines = this.savedProgram.toLowerCase().split('\n')
      let take = false
      let found = false
      const out = []
      lines.forEach(function (l) {
        if (/^\s*#/.test(l)) {
          take = /(worth|not yet|untried|to try|explore|idea)/.test(l)
          if (take) found = true
        } else if (take) {
          out.push(l)
        }
      })
      return found ? out.join('\n') : lines.join('\n')
    },
    createTrial() {
      const n = this.trials.length + 1
      const rank = this.rankCandidates()
      this.trials.push({
        n: n,
        stage: 'propose',
        node: 'propose_trial',
        t: 0,
        done: false,
        final: '',
        cand: rank[(n - 1) % rank.length],
        patchText: '',
        patchReady: false,
        pairs: [],
        removed: [],
        decided: false,
        decisionText: '',
        bestBefore: BASELINE,
        reflected: false,
        noted: false,
        earlyFile: false,
        early: false,
        debugNote: '',
        logLines: [],
        job: '',
        ready: false,
        outcome: '',
        attempt: 1,
        progress: 0,
        epoch: 0,
        loss: [],
        score: null,
        plan: { fail: n === 3, early: n === 5 }
      })
    },
    setStage(tr, stage) {
      tr.stage = stage
      tr.node = STAGE_NODE[stage]
      tr.t = 0
      tr.ready = false
    },
    submit(tr) {
      this.jobSeq += 1
      tr.job = 'job-' + pad(this.jobSeq, 4)
      if (tr.attempt > 1) tr.logLines.push('--- attempt ' + tr.attempt + ' ---')
      tr.logLines.push('[' + tr.job + '] submitted')
      this.acquireGpu(tr)
      this.setStage(tr, 'pending')
    },
    lossAt(tr, e) {
      let v = 0.9 + 2.3 * Math.exp(-0.45 * e) + (this.rng() - 0.5) * 0.06
      if (tr.plan.early && e > 3) v += 0.4 * (e - 3)
      return v
    },
    work(tr, dt) {
      tr.t += dt
      if (tr.stage === 'pending') {
        if (tr.t >= DUR.pending) {
          this.setStage(tr, 'train')
          this.gpus.forEach(function (g) { if (g.trial === tr.n) g.state = 'running' })
          tr.logLines.push('[' + tr.job + '] RUNNING')
        }
        return
      }
      if (tr.stage === 'train') {
        if (!tr.ready) this.advanceTrain(tr)
        return
      }
      if (!tr.ready && tr.t >= WORK[tr.stage]) tr.ready = true
    },
    step(tr) {
      const s = tr.stage
      if (s === 'propose') return this.finishPropose(tr)
      if (s === 'train') {
        if (tr.outcome === 'failed') {
          tr.debugNote = '# debug_trial\n\nFailure: CUDA out of memory (attempt 1)\nFix: lower dataloader.batch_bins 16000000 -> 8000000, then rerun.\n'
          return this.setStage(tr, 'debug')
        }
        return this.setStage(tr, 'collect')
      }
      if (s === 'debug') {
        tr.pairs.push(['dataloader.batch_bins', '8000000'])
        tr.attempt = 2
        tr.progress = 0
        tr.epoch = 0
        tr.loss = []
        tr.outcome = ''
        tr.plan.fail = false
        return this.submit(tr)
      }
      if (s === 'collect') return this.finishCollect(tr)
      if (s === 'decide') return this.finishDecide(tr)
      if (s === 'reflect') {
        tr.reflected = true
        return this.setStage(tr, 'knowledge')
      }
      if (s === 'knowledge') {
        tr.noted = true
        return this.setStage(tr, 'budget')
      }
      if (s === 'budget') tr.done = true
    },
    finishPropose(tr) {
      let pairs = tr.cand.pairs.slice()
      if (tr.n === 2) pairs = pairs.concat([['dataset.test_sets', 'test-other']])
      const kept = []
      const removed = []
      pairs.forEach(function (p) {
        const deny = DENIED.some(function (d) { return p[0].indexOf(d) === 0 })
        const allow = ALLOWED.some(function (a) { return p[0].indexOf(a) === 0 })
        if (deny || !allow) removed.push(p[0])
        else kept.push(p)
      })
      tr.pairs = kept
      tr.removed = removed
      tr.patchText = kept.map(function (p) { return p[0].split('.').slice(-1)[0] + '=' + p[1] }).join(', ')
      tr.patchReady = true
      this.submit(tr)
    },
    advanceTrain(tr) {
      tr.progress = Math.min(1, tr.t / DUR.train)
      const ep = Math.floor(tr.progress * EPOCHS)
      while (tr.epoch < ep) {
        tr.epoch += 1
        const v = this.lossAt(tr, tr.epoch)
        tr.loss.push(v)
        tr.logLines.push('epoch ' + tr.epoch + '/' + EPOCHS + '  loss=' + v.toFixed(3))
      }
      if (tr.plan.fail && tr.progress >= 0.35) {
        tr.logLines.push('RuntimeError: CUDA out of memory')
        tr.logLines.push('[' + tr.job + '] FAILED')
        this.releaseGpu(tr)
        tr.outcome = 'failed'
        tr.ready = true
        return
      }
      if (tr.plan.early && tr.epoch >= 5) {
        tr.logLines.push('monitor: loss rising since epoch 3 -> EARLY_STOP_REQUESTED')
        tr.logLines.push('[' + tr.job + '] stopped early')
        tr.earlyFile = true
        tr.early = true
        this.releaseGpu(tr)
        tr.outcome = 'early'
        tr.ready = true
        return
      }
      if (tr.progress >= 1) {
        tr.logLines.push('[' + tr.job + '] COMPLETED')
        this.releaseGpu(tr)
        tr.outcome = 'done'
        tr.ready = true
      }
    },
    finishCollect(tr) {
      if (!tr.early) {
        const jitter = (this.rng() - 0.5) * 0.06
        tr.score = Math.round((BASELINE + tr.cand.delta + jitter) * 100) / 100
      }
      this.setStage(tr, 'decide')
    },
    finishDecide(tr) {
      tr.bestBefore = this.best
      if (tr.early) {
        tr.final = 'early_stopped'
      } else if (tr.score < this.best) {
        tr.final = 'accepted'
        this.best = tr.score
      } else {
        tr.final = 'rejected'
      }
      tr.decisionText = '# Decision\n\ndev_wer: ' + (tr.score === null ? 'n/a (stopped early)' : tr.score.toFixed(2)) + '\nbest_before: ' + tr.bestBefore.toFixed(2) + '\nverdict: ' + tr.final + '\n'
      tr.decided = true
      this.setStage(tr, 'reflect')
    }
  }
}
</script>

<style>
.atlasd {
  --ad-bg: #eef1f5;
  --ad-surface: #ffffff;
  --ad-ink: #17202b;
  --ad-muted: #5b6776;
  --ad-line: #d5dbe3;
  --ad-agent: #6247d9;
  --ad-agent-tint: #ece8fb;
  --ad-gpu: #b35c00;
  --ad-gpu-tint: #fbefdc;
  --ad-ok: #1b8a57;
  --ad-bad: #c4382f;
  --ad-term-bg: #121821;
  --ad-term-ink: #d7dee8;
  --ad-term-dim: #7f8c9d;
  --ad-h: 340px;
  container-type: inline-size;
  margin: 1.5rem 0;
  padding: 16px;
  border: 1px solid var(--ad-line);
  border-radius: 10px;
  background: var(--ad-bg);
  color: var(--ad-ink);
  font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;
  font-size: 14px;
  line-height: 1.5;
}
html.dark .atlasd,
[data-theme='dark'] .atlasd {
  --ad-bg: #0f141b;
  --ad-surface: #171e28;
  --ad-ink: #e6ebf2;
  --ad-muted: #93a0b2;
  --ad-line: #2a3442;
  --ad-agent: #a595ff;
  --ad-agent-tint: #262244;
  --ad-gpu: #f0a04b;
  --ad-gpu-tint: #3a2a14;
  --ad-ok: #4cc38a;
  --ad-bad: #ff7a70;
}
.atlasd * {
  box-sizing: border-box;
}
.atlasd-note {
  margin: 0 0 12px;
  color: var(--ad-muted);
  font-size: 13px;
}
.atlasd-steps {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 18px;
  margin: 0 0 12px;
  padding: 0;
  list-style: none;
  color: var(--ad-muted);
}
.atlasd-steps li {
  display: flex;
  align-items: center;
  gap: 6px;
}
.atlasd-step-n {
  display: inline-grid;
  place-items: center;
  width: 20px;
  height: 20px;
  border: 1px solid var(--ad-line);
  border-radius: 50%;
  font-size: 12px;
}
.atlasd-steps li.now {
  color: var(--ad-ink);
  font-weight: 600;
}
.atlasd-steps li.now .atlasd-step-n {
  border-color: var(--ad-ink);
}
.atlasd-steps li.done .atlasd-step-n {
  background: var(--ad-ok);
  border-color: var(--ad-ok);
  color: #fff;
}
.atlasd-cmds {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  margin-bottom: 14px;
}
.atlasd-spacer {
  flex: 1 1 auto;
}
.atlasd-btn {
  padding: 7px 14px;
  border: 1px solid var(--ad-line);
  border-radius: 6px;
  background: var(--ad-surface);
  color: var(--ad-ink);
  font: 600 13px/1.2 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  cursor: pointer;
}
.atlasd-btn:hover:not(:disabled) {
  border-color: var(--ad-ink);
}
.atlasd-btn:focus-visible,
.atlasd-chip:focus-visible,
.atlasd-editor:focus-visible,
.atlasd-tree li.link:focus-visible {
  outline: 2px solid var(--ad-agent);
  outline-offset: 2px;
}
.atlasd-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}
.atlasd-btn.is-next {
  background: var(--ad-ink);
  border-color: var(--ad-ink);
  color: var(--ad-surface);
}
.atlasd-btn-stop:not(:disabled) {
  border-color: var(--ad-bad);
  color: var(--ad-bad);
}
.atlasd-btn-quiet {
  font-family: inherit;
  font-weight: 500;
  color: var(--ad-muted);
}
.atlasd-speed {
  display: inline-flex;
  align-items: center;
  gap: 4px;
}
.atlasd-speed-label {
  margin-right: 4px;
  color: var(--ad-muted);
  font-size: 12px;
}
.atlasd-chip {
  padding: 4px 8px;
  border: 1px solid var(--ad-line);
  border-radius: 4px;
  background: transparent;
  color: var(--ad-muted);
  font-size: 12px;
  cursor: pointer;
}
.atlasd-chip.on {
  border-color: var(--ad-ink);
  color: var(--ad-ink);
}
.atlasd-row {
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: 14px;
  margin-bottom: 14px;
}
@container (min-width: 820px) {
  .atlasd-row {
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  }
  .atlasd .atlasd-trials {
    grid-template-columns: minmax(0, 2fr) minmax(0, 3fr);
  }
  .atlasd .atlasd-trials-chart {
    border-right: 1px solid var(--ad-line);
  }
}
.atlasd-card {
  border: 1px solid var(--ad-line);
  border-radius: 8px;
  background: var(--ad-surface);
  overflow: hidden;
}
.atlasd-head {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 4px 12px;
  padding: 8px 12px;
  border-bottom: 1px solid var(--ad-line);
  font-size: 13px;
}
.atlasd-head-hint {
  margin-left: auto;
  color: var(--ad-muted);
  font-size: 12px;
}
.atlasd-fname {
  font: 700 13px/1.4 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  word-break: break-all;
}
.atlasd-viewer {
  display: flex;
  flex-direction: column;
  height: var(--ad-h);
}
.atlasd-editor {
  display: block;
  flex: 1 1 auto;
  width: 100%;
  min-height: 0;
  margin: 0;
  padding: 10px 12px;
  border: 0;
  background: transparent;
  color: var(--ad-ink);
  font: 12.5px/1.6 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  resize: none;
  overflow: auto;
  white-space: pre;
}
.atlasd-placeholder {
  display: grid;
  flex: 1 1 auto;
  place-content: center;
  padding: 16px;
  color: var(--ad-muted);
  text-align: center;
}
.atlasd-placeholder p {
  margin: 0;
  max-width: 28ch;
}
.atlasd-placeholder code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  color: var(--ad-ink);
}
.atlasd-term {
  height: var(--ad-h);
  padding: 10px 12px;
  border-radius: 8px;
  background: var(--ad-term-bg);
  color: var(--ad-term-ink);
  font: 12px/1.55 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  overflow: auto;
}
.atlasd-line {
  white-space: pre-wrap;
  word-break: break-word;
}
.atlasd-line.k-cmd {
  margin-top: 6px;
  color: #fff;
  font-weight: 700;
}
.atlasd-line.k-cmd:first-child {
  margin-top: 0;
}
.atlasd-line.k-dim {
  color: var(--ad-term-dim);
}
.atlasd-treecard {
  display: flex;
  flex-direction: column;
}
.atlasd-tree {
  flex: 1 1 auto;
  max-height: 330px;
  min-height: 160px;
  margin: 0;
  padding: 6px 0;
  list-style: none;
  font: 12px/1.65 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  overflow: auto;
}
.atlasd-tree li {
  padding-right: 12px;
}
.atlasd-tree li.dir {
  font-weight: 700;
}
.atlasd-tree li.file {
  color: var(--ad-muted);
}
.atlasd-tree li.link {
  cursor: pointer;
}
.atlasd-tree li.link:hover {
  background: var(--ad-bg);
  color: var(--ad-ink);
}
.atlasd-tree li.special {
  color: var(--ad-bad);
  font-weight: 700;
}
.atlasd-tree li.sel {
  background: var(--ad-agent-tint);
  color: var(--ad-ink);
  box-shadow: inset 3px 0 0 var(--ad-agent);
}
.atlasd-tree-enter-active {
  transition: opacity 0.4s, transform 0.4s;
}
.atlasd-tree-enter,
.atlasd-tree-enter-from {
  opacity: 0;
  transform: translateX(-6px);
}
.atlasd-legend {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  margin-left: auto;
  color: var(--ad-muted);
  font-size: 12px;
}
.atlasd-legend .sw {
  display: inline-block;
  width: 12px;
  height: 12px;
  margin-left: 6px;
  border: 1.5px solid var(--ad-line);
  border-radius: 3px;
}
.atlasd-legend .sw-agent {
  background: var(--ad-agent-tint);
  border-color: var(--ad-agent);
}
.atlasd-graph {
  display: block;
  width: 100%;
  height: auto;
  padding: 6px 8px 10px;
}
.atlasd-edge {
  fill: none;
  stroke: var(--ad-muted);
  stroke-width: 1.4;
}
.atlasd-edge.dashed {
  stroke-dasharray: 4 3;
}
.atlasd-arrowhead {
  fill: var(--ad-muted);
}
.atlasd-edge-label {
  fill: var(--ad-muted);
  font-size: 10px;
}
.atlasd-node rect {
  fill: var(--ad-surface);
  stroke: var(--ad-line);
  stroke-width: 1.5;
  transition: fill 0.3s, stroke 0.3s;
}
.atlasd-node.is-agent rect {
  fill: var(--ad-agent-tint);
  stroke: var(--ad-agent);
  stroke-opacity: 0.45;
}
.atlasd-node text {
  fill: var(--ad-ink);
  font: 600 10.5px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
.atlasd-node.is-active rect {
  stroke-width: 2.5;
  stroke-opacity: 1;
  stroke: var(--ad-ink);
}
.atlasd-node.is-agent.is-active rect {
  stroke: var(--ad-agent);
}
.atlasd-node.is-gpu rect {
  fill: var(--ad-gpu-tint);
  stroke: var(--ad-gpu);
}
.atlasd-token {
  transition: transform 0.6s cubic-bezier(0.4, 0, 0.2, 1);
}
.atlasd-wait {
  fill: none;
  stroke: var(--ad-ink);
  stroke-width: 1.5;
  stroke-dasharray: 3 2.5;
}
.atlasd-tickrow {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 12px 0;
  color: var(--ad-muted);
  font-size: 12px;
}
.atlasd-tickrow.off {
  opacity: 0.5;
}
.atlasd-tickbar {
  flex: 0 0 90px;
  height: 6px;
  border-radius: 3px;
  background: var(--ad-line);
  overflow: hidden;
}
.atlasd-tickbar b {
  display: block;
  height: 100%;
  background: var(--ad-ink);
}
.atlasd-tickhint {
  margin-left: auto;
}
.atlasd-token circle {
  stroke: var(--ad-surface);
  stroke-width: 2;
}
.atlasd-token text {
  fill: #fff;
  font: 700 11px system-ui, sans-serif;
}
.atlasd-gpucard {
  margin-bottom: 14px;
}
.atlasd-gpus {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
}
.atlasd-gpu {
  padding: 10px 12px;
  border-right: 1px solid var(--ad-line);
}
.atlasd-gpu:last-child {
  border-right: 0;
}
.atlasd-gpu-top {
  display: flex;
  justify-content: space-between;
  font-weight: 600;
}
.atlasd-gpu-state {
  color: var(--ad-muted);
  font-size: 12px;
  font-weight: 500;
}
.atlasd-gpu.s-running .atlasd-gpu-state {
  color: var(--ad-gpu);
  font-weight: 700;
}
.atlasd-gpu-trial {
  min-height: 20px;
  color: var(--ad-muted);
  font-size: 12px;
}
.atlasd-meter {
  display: grid;
  grid-template-columns: 30px 1fr 34px;
  align-items: center;
  gap: 6px;
  margin-top: 4px;
  color: var(--ad-muted);
  font-size: 11px;
}
.atlasd-meter .bar {
  height: 6px;
  border-radius: 3px;
  background: var(--ad-line);
  overflow: hidden;
}
.atlasd-meter .bar b {
  display: block;
  height: 100%;
  background: var(--ad-gpu);
  transition: width 0.15s linear;
}
.atlasd-meter .val {
  text-align: right;
}
.atlasd-spark {
  display: block;
  width: 100%;
  height: 36px;
  margin-top: 8px;
  border-bottom: 1px solid var(--ad-line);
}
.atlasd-spark polyline {
  fill: none;
  stroke: var(--ad-ink);
  stroke-width: 1.6;
  vector-effect: non-scaling-stroke;
}
.atlasd-trials {
  display: grid;
  grid-template-columns: minmax(0, 1fr);
}
.atlasd-chart {
  display: block;
  width: 100%;
  height: auto;
  padding: 4px 12px 8px;
}
.atlasd-base {
  stroke: var(--ad-muted);
  stroke-dasharray: 3 3;
}
.atlasd-base-label {
  fill: var(--ad-muted);
  font-size: 6.5px;
}
.atlasd-bestline {
  fill: none;
  stroke: var(--ad-ok);
  stroke-width: 1.6;
}
.atlasd-chart .pt-ok {
  fill: var(--ad-ok);
}
.atlasd-chart .pt-no {
  fill: var(--ad-muted);
}
.atlasd-table {
  align-self: start;
  width: 100%;
  margin: 0;
  border-collapse: collapse;
  font-size: 12.5px;
}
.atlasd-table td {
  padding: 6px 12px;
  border-top: 1px solid var(--ad-line);
  vertical-align: middle;
}
.atlasd-table tr:first-child td {
  border-top: 0;
}
.atlasd-table .c-n {
  width: 44px;
  white-space: nowrap;
}
.atlasd-table .dot {
  display: inline-block;
  width: 9px;
  height: 9px;
  margin-right: 6px;
  border-radius: 50%;
}
.atlasd-table .c-patch {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 11.5px;
  word-break: break-all;
}
.atlasd-table .c-score {
  width: 48px;
  text-align: right;
  font-variant-numeric: tabular-nums;
}
.atlasd-pill {
  display: inline-block;
  padding: 1px 8px;
  border: 1px solid var(--ad-line);
  border-radius: 10px;
  color: var(--ad-muted);
  font-size: 11.5px;
  white-space: nowrap;
}
.atlasd-pill.p-accepted {
  border-color: var(--ad-ok);
  color: var(--ad-ok);
  font-weight: 700;
}
.atlasd-pill.p-early_stopped {
  border-color: var(--ad-bad);
  color: var(--ad-bad);
}
.atlasd-empty {
  padding: 10px 12px;
  color: var(--ad-muted);
  font-size: 12.5px;
}
@media (prefers-reduced-motion: reduce) {
  .atlasd-token,
  .atlasd-tree-enter-active,
  .atlasd-meter .bar b,
  .atlasd-node rect {
    transition: none;
  }
}
</style>
