// Cookie-owned bounded Table.action service. No Nodus routing IDs grant authority.
// Explicit recover() resends the identical sequence after a lost response; the
// server returns its receipt, never repeats an old/evicted/changed-epoch batch.
export class ActionBatchClient {
  constructor(fetcher = globalThis.fetch.bind(globalThis)) {
    this.fetcher = fetcher;
    this.meta = null;
    this.pending = null;
    this.busy = false;
  }
  async request(body) {
    const response = await this.fetcher('/api/action/batch', {
      method: 'POST', credentials: 'same-origin',
      headers: {'Content-Type': 'application/json', 'X-Meowdoku-Batch': '1'},
      body: JSON.stringify(body),
    });
    const value = await response.json();
    if (!response.ok) throw new Error(value.error || `Batch HTTP ${response.status}`);
    return value;
  }
  async describe() {
    if (this.busy || this.pending) throw new Error('Resolve pending batch before changing identity');
    this.busy = true;
    try { this.meta = await this.request({op: 'describe'}); return this.meta; }
    finally { this.busy = false; }
  }
  async run(actions) {
    if (!this.meta || this.busy || this.pending) throw new Error('Describe or resolve the pending batch first');
    if (!Array.isArray(actions) || actions.length < 1 || actions.length > 32) throw new Error('Use 1 to 32 actions');
    this.pending = JSON.parse(JSON.stringify({table_epoch: this.meta.table_epoch,
      sequence: this.meta.next_sequence, actions}));
    return this.recover();
  }
  async recover() {
    if (this.busy || !this.pending) throw new Error('No recoverable batch or request already running');
    this.busy = true;
    try {
      const receipt = await this.request(this.pending);
      if (receipt.table_epoch !== this.pending.table_epoch || receipt.sequence !== this.pending.sequence)
        throw new Error('Mismatched receipt; inspect state before proceeding');
      this.meta.next_sequence = receipt.next_sequence;
      this.pending = null;
      return receipt; // stopped includes a possibly-mutated failed action: never retry its suffix automatically
    } finally { this.busy = false; }
  }
}
