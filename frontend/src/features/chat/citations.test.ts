import { describe, expect, it } from "vitest";

import { linkCitations } from "./citations";

describe("linkCitations", () => {
  it("turns citation markers into links", () => {
    expect(linkCitations("Leave is 18 days [1].")).toBe("Leave is 18 days [1](#cite-1).");
  });

  it("splits grouped citations", () => {
    expect(linkCitations("See [1, 3]")).toBe("See [1](#cite-1)[3](#cite-3)");
  });

  it("leaves existing Markdown links and plain brackets alone", () => {
    expect(linkCitations("[2](https://example.com) and [note]")).toBe(
      "[2](https://example.com) and [note]",
    );
  });
});
