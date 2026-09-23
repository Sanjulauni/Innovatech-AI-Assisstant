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

  it("handles full-width citations from models like GPT-OSS", () => {
    const L = "【";
    const R = "】";
    expect(linkCitations(`Creative Commons ${L}1${R} ${L}2${R}`)).toBe(
      "Creative Commons [1](#cite-1) [2](#cite-2)",
    );
    expect(linkCitations(`See ${L}1， 3${R}`)).toBe("See [1](#cite-1)[3](#cite-3)");
    expect(linkCitations(`Note ${L}4†L1-L5${R}`)).toBe("Note [4](#cite-4)");
    expect(linkCitations(`Keep ${L}note${R}`)).toBe(`Keep ${L}note${R}`);
  });
});
