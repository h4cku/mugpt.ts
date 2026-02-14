export class Tokenizer {
  BOS: number;
  vocabSize: number;
  idx2char: string[];

  constructor(docs: string[]) {
    this.idx2char = Array.from(new Set(docs.join(""))).sort();
    this.BOS = this.idx2char.length;
    this.vocabSize = this.idx2char.length + 1;
    console.log("vocab size:", this.vocabSize);
  }
}
