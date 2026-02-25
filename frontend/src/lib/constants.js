// Shared display mappings used across multiple components

export const MODEL_SHORT = {
  'claude-haiku-4-5-20251001':  'Haiku 4.5',
  'claude-sonnet-4-5-20250929': 'Sonnet 4.5',
  'claude-opus-4-6':            'Opus 4.6',
}

export const MODEL_COLOR = {
  'claude-haiku-4-5-20251001':  '#10b981', // emerald-500
  'claude-sonnet-4-5-20250929': '#3b82f6', // blue-500
  'claude-opus-4-6':            '#a855f7', // purple-500
}

export const MODEL_PILL = {
  'claude-haiku-4-5-20251001':  'bg-emerald-500/20 text-emerald-300',
  'claude-sonnet-4-5-20250929': 'bg-blue-500/20    text-blue-300',
  'claude-opus-4-6':            'bg-purple-500/20  text-purple-300',
}

export const TASK_LABEL = {
  simple_qa:        'Simple Q&A',
  summarization:    'Summarization',
  code_generation:  'Code Gen',
  creative_writing: 'Creative',
  math_reasoning:   'Math',
  analysis:         'Analysis',
}

export const TASK_COLOR = {
  simple_qa:        '#06b6d4', // cyan-500
  summarization:    '#f59e0b', // amber-500
  code_generation:  '#10b981', // emerald-500
  creative_writing: '#ec4899', // pink-500
  math_reasoning:   '#f97316', // orange-500
  analysis:         '#6366f1', // indigo-500
}

export const TASK_PILL = {
  simple_qa:        'bg-cyan-500/20    text-cyan-300',
  summarization:    'bg-amber-500/20   text-amber-300',
  code_generation:  'bg-emerald-500/20 text-emerald-300',
  creative_writing: 'bg-pink-500/20    text-pink-300',
  math_reasoning:   'bg-orange-500/20  text-orange-300',
  analysis:         'bg-indigo-500/20  text-indigo-300',
}
