const CITATION = /\[(\d+(?:\s*,\s*\d+)*)\](?!\()/g;
// Some models cite with full-width brackets: 【1】, 【1, 2】 or 【1†source】.
const WIDE_CITATION = /【\s*(\d+(?:\s*[,，]\s*\d+)*)[^】]*】/g;
const COMMA = /\s*[,，]\s*/;

export const CITE_PREFIX = "#cite-";

const toLinks = (group: string) =>
  group
    .split(COMMA)
    .map((n) => `[${n.trim()}](${CITE_PREFIX}${n.trim()})`)
    .join("");

/** Turn "[1]", "[1, 2]" and "【1】" into links that the renderer shows as citation chips. */
export function linkCitations(text: string): string {
  return text
    .replace(WIDE_CITATION, (_match, group: string) => toLinks(group))
    .replace(CITATION, (_match, group: string) => toLinks(group));
}
