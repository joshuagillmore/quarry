Quarry stores every crawled page as markdown, and this fixture stands in for one of those pages. It opens with two paragraphs under no heading at all, because a real page has text before its first heading and that text must land in a chunk with no section prefix.

The chunker is shared with Coherence in spirit and in rule: paragraphs are atomic, headings set the section, and a chunk never straddles two sections. This second introductory paragraph is here so that the first chunk holds more than one span.

# Why chunk size is a corpus property

The original nine-hundred-character target was set against a corpus of short exercise reports where almost every document fit in one chunk. Against long-form reporting the same target binds on every document and cuts the median report into fifteen fragments, which destroys exactly what the atomicity rule exists to protect.

So the target is a policy supplied by the caller, and different corpora get different ones. Two thousand characters is about five hundred tokens, large enough to carry a complete argument and small enough that one embedding is not averaged across several.

The following paragraph is deliberately long, well past the two-thousand-character target, so that the fixture proves a single paragraph is never split even when it alone exceeds the target. A citation into a chunk must resolve to a passage an analyst can read and check in full, and a paragraph cut in half is a chunk whose second part has lost the argument it belongs to. The sentence before this one is the whole justification; what follows is filler of the ordinary kind, the sort of prose a crawled page carries in abundance, written so the paragraph can reach its length without repeating a single sentence verbatim. Consider the way a long report describes its own method: it names the sources it drew on, the period they cover, the questions it put to them, and the limits it placed on its own conclusions. It then walks through each finding in turn, pausing to note where two sources disagree and how the disagreement was resolved, or left open with a stated reason. It closes each section with what remains unknown, because a reader who is told only what is known cannot judge how much weight the finding deserves. All of that is one continuous passage in the original, and a reader who meets only its back half, stripped of the method that opened it, has no way to tell a careful finding from a confident guess. The same holds for a page that explains a procedure step by step: the order of the steps is the content, and a chunk that begins at step four carries the words of the procedure without its meaning. It holds again for a page that argues a position: the premises come first, the conclusion last, and the conclusion alone is an assertion the reader cannot check. For a transcript the speaker turns are the structure; for a table rendered as text the header row is the key to every row below it; for a legal clause the defined terms at the top govern every sentence after them. In each case the boundary that matters is the one the author drew, and the chunker's only job at that boundary is to respect it. The size rule exists for the case where the author drew no boundary for a long time, and even there it waits for the next one rather than inventing its own, which is why this paragraph, long as it is, must arrive in a chunk by itself and whole.

## Section boundaries are chunk boundaries ##

A chunk spanning two sections carries two topics. Its embedding is the average of both and its section attribution becomes a lie, so a heading always closes the open chunk, even when that chunk is far short of the target.

The heading text is prepended to every chunk under it. A mid-document paragraph reading that the committee may wish to consider whether current authorities are adequate is nearly contentless on its own; under its heading it is retrievable.

What the port adds for markdown is a parse step, and this list enumerates its span types:

- a paragraph, meaning a run of non-blank lines
- a single list item, kept whole with any continuation lines
  that are indented beneath it
- a fenced code block, kept whole even when it contains blank lines
- never a heading line, which sets the section and is not itself a span

Each list item is its own span, which is how Coherence's HTML parser treats a list element, so a long list can be divided at an item boundary without ever cutting through one.

The size rule is the third and last of the boundary tests, and it is the only one that is a size decision rather than a correctness decision. When the open chunk already holds something and the next span would take it past the target, the chunk is closed and the span opens a new one under the same section. The span is never trimmed to fit; it moves whole. This paragraph and the next exist so that this section grows past the target and that rule fires here, between two ordinary paragraphs, rather than only on the oversize paragraph above.

A chunk closed by size keeps its section prefix, and the chunk opened after it gets the same prefix again, so both remain retrievable under the heading they belong to. Nothing about the text changes when the boundary falls; the only effect is where the embedding averages stop and start. That is the whole of the rule, and the fixture now covers each of its three terms in order: a section change, the kind term that markdown never exercises, and the target being reached.

### Code and noise

A fenced block is one span. The blank line inside the block below must not end it:

```python
def indexed_text(title, text):

    return f"{title}\n\n{text}"
```

the and of to a in

That last paragraph is stopwords only. The chunker does not judge content; it is a boundary rule, and page quality is decided elsewhere, before the page is ever stored.

A closing paragraph under the third heading, short, so that the final chunk ends the document with a section prefix and a few spans behind it.
