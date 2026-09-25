import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import {
  callsTo,
  HEALTH,
  jsonResponse,
  mockApi,
  ndjson,
  renderApp,
  streamResponse,
} from "../../test/utils";

const SOURCE = {
  index: 1,
  doc_id: "abc",
  source: "Leave_Policy.txt",
  page: null,
  snippet: "Full-time employees receive 18 days of paid annual leave.",
  score: 0.8,
};

const answerStream = (answer = "You get **18 days** of leave [1].") =>
  streamResponse(
    ndjson(
      { type: "token", text: answer.slice(0, 10) },
      { type: "token", text: answer.slice(10) },
      { type: "done", answer, sources: [SOURCE] },
    ),
  );

async function ask(question: string) {
  const user = userEvent.setup();
  await user.type(screen.getByLabelText("Your question"), `${question}{Enter}`);
  return user;
}

describe("ChatPage", () => {
  it("shows suggestions when empty and asks one when clicked", async () => {
    const fetchMock = mockApi({ "POST /chat/stream": () => answerStream() });
    renderApp();

    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "How many days of annual leave do I get?" }));

    expect(await screen.findByText("18 days")).toBeInTheDocument();
    expect(callsTo(fetchMock, "POST", "/chat/stream")[0]).toMatchObject({
      question: "How many days of annual leave do I get?",
    });
  });

  it("streams the answer, renders Markdown and shows sources on citation click", async () => {
    mockApi({ "POST /chat/stream": () => answerStream() });
    renderApp();

    const user = await ask("How much leave?");

    expect(screen.getByText("How much leave?")).toBeInTheDocument();
    const bold = await screen.findByText("18 days");
    expect(bold.tagName).toBe("STRONG");

    // Sources are collapsed until asked for.
    expect(screen.queryByText("Leave_Policy.txt")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Show source 1" }));
    expect(screen.getByText("Leave_Policy.txt")).toBeInTheDocument();
    expect(screen.getByText(SOURCE.snippet)).toBeInTheDocument();
  });

  it("makes full-width citations clickable while streaming", async () => {
    // The stream ends without a "done" event, so only the raw text is shown.
    mockApi({
      "POST /chat/stream": () =>
        streamResponse(ndjson({ type: "token", text: "The company is Creative Commons 【1】" })),
    });
    renderApp();

    await ask("Which company?");

    expect(await screen.findByRole("button", { name: "Show source 1" })).toBeInTheDocument();
    expect(screen.queryByText(/【/)).not.toBeInTheDocument();
  });

  it("sends earlier turns as history", async () => {
    const fetchMock = mockApi({ "POST /chat/stream": () => answerStream() });
    renderApp();

    await ask("How much leave?");
    await screen.findByText("18 days");
    await ask("And for part-time staff?");

    await waitFor(() => expect(callsTo(fetchMock, "POST", "/chat/stream")).toHaveLength(2));
    expect(callsTo(fetchMock, "POST", "/chat/stream")[1]).toEqual({
      question: "And for part-time staff?",
      history: [
        { role: "user", content: "How much leave?" },
        { role: "assistant", content: "You get **18 days** of leave [1]." },
      ],
    });
  });

  it("does not send empty questions", async () => {
    const fetchMock = mockApi({});
    renderApp();

    await ask("   ");

    expect(screen.getByRole("button", { name: "Send question" })).toBeDisabled();
    expect(callsTo(fetchMock, "POST", "/chat/stream")).toHaveLength(0);
  });

  it("shows a friendly error and retries", async () => {
    let calls = 0;
    mockApi({
      "POST /chat/stream": () =>
        ++calls === 1
          ? jsonResponse({ detail: "The AI model is busy (usage limit reached)." }, 503)
          : answerStream(),
    });
    renderApp();

    const user = await ask("How much leave?");

    expect(await screen.findByRole("alert")).toHaveTextContent("usage limit reached");
    await user.click(screen.getByRole("button", { name: "Try again" }));

    expect(await screen.findByText("18 days")).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.getAllByText("How much leave?")).toHaveLength(1);
  });

  it("shows an error event that arrives mid-answer", async () => {
    mockApi({
      "POST /chat/stream": () =>
        streamResponse(
          ndjson({ type: "token", text: "Partial" }, { type: "error", message: "Model went away." }),
        ),
    });
    renderApp();

    await ask("Question?");

    expect(await screen.findByRole("alert")).toHaveTextContent("Model went away.");
  });

  it("starts a new chat and remembers the chat across reloads", async () => {
    mockApi({ "POST /chat/stream": () => answerStream() });
    const first = renderApp();
    await ask("How much leave?");
    await screen.findByText("18 days");

    first.unmount();
    renderApp();
    expect(screen.getByText("How much leave?")).toBeInTheDocument();

    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: /new chat/i }));
    expect(screen.queryByText("How much leave?")).not.toBeInTheDocument();
    expect(screen.getByText("How can I help?")).toBeInTheDocument();
  });

  it.each([
    ["local", true],
    ["groq", false],
  ])("while the %s model reads the question, explains the wait: %s", async (provider, shown) => {
    let finish = () => {};
    mockApi({
      "GET /health": () => jsonResponse({ ...HEALTH, llm_provider: provider }),
      "POST /chat/stream": () =>
        new Promise<Response>((resolve) => {
          finish = () => resolve(answerStream());
        }),
    });
    renderApp();
    await screen.findByTitle(/Model:/); // health has loaded
    await ask("How much leave?");

    expect(await screen.findByText("Searching the documents…")).toBeInTheDocument();
    const hint = screen.queryByText(/first words can take about a minute/);
    expect(hint !== null).toBe(shown);

    finish();
    expect(await screen.findByText("18 days")).toBeInTheDocument();
    expect(screen.queryByText(/first words can take about a minute/)).not.toBeInTheDocument();
  });

  it("shows server status in the header", async () => {
    mockApi({});
    renderApp();
    const header = screen.getByRole("banner");
    expect(await within(header).findByText("2 documents")).toBeInTheDocument();
  });
});
