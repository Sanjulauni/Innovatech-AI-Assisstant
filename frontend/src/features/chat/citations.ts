const CITATION = /\[(\d+(?:\s*,\s*\d+)*)\](?!\()/g;
export const CITE_PREFIX = "#cite-";

/** Turn "[1]" and "[1, 2]" into links that the renderer shows as citation chips. */
export function linkCitations(text: string): string {
  return text.replace(CITATION, (_match, group: string) =>
    group
      .split(",")
      .map((n) => `[${n.trim()}](${CITE_PREFIX}${n.trim()})`)
      .join(""),
  );
}
