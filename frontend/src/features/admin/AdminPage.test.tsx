import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { callsTo, HEALTH, jsonResponse, mockApi, renderApp } from "../../test/utils";

const DOC = {
  doc_id: "a".repeat(64),
  source: "Leave_Policy.txt",
  chunk_count: 3,
  ingested_at: "2026-09-23T08:00:00+00:00",
};

const LIMIT = { max_upload_size_mb: 20, default_mb: 20, max_allowed_mb: 200 };

const unauthorized = () => jsonResponse({ detail: "Incorrect admin password." }, 401);

/** Routes for a logged-in admin with the given documents. */
function loggedIn(documents = [DOC], extra = {}) {
  return mockApi({
    "GET /admin/session": () => jsonResponse({ authenticated: true }),
    "GET /admin/documents": () => jsonResponse(documents),
    "GET /admin/instructions": () => jsonResponse({ text: "Be concise.", updated_at: null }),
    "GET /admin/upload-limit": () => jsonResponse(LIMIT),
    ...extra,
  });
}

describe("AdminPage sign in", () => {
  it("asks for the password and signs in", async () => {
    let authenticated = false;
    const fetchMock = mockApi({
      "GET /admin/session": () => (authenticated ? jsonResponse({ authenticated }) : unauthorized()),
      "POST /admin/login": (_url, init) => {
        const { password } = JSON.parse(init.body as string);
        if (password !== "secret") return unauthorized();
        authenticated = true;
        return jsonResponse({ authenticated: true, expires_in: 28800 });
      },
      "GET /admin/documents": () => jsonResponse([DOC]),
      "GET /admin/upload-limit": () => jsonResponse(LIMIT),
    });
    renderApp("/admin");
    const user = userEvent.setup();

    const password = await screen.findByLabelText("Password");
    await user.type(password, "wrong{Enter}");
    expect(await screen.findByRole("alert")).toHaveTextContent("Incorrect admin password.");

    await user.clear(password);
    await user.type(password, "secret{Enter}");

    expect(await screen.findByText("Leave_Policy.txt")).toBeInTheDocument();
    expect(callsTo(fetchMock, "POST", "/admin/login")).toEqual([
      { password: "wrong" },
      { password: "secret" },
    ]);
  });

  it("explains when admin is disabled on the server", async () => {
    mockApi({
      "GET /admin/session": () => jsonResponse({ detail: "Admin access is disabled." }, 503),
      "GET /health": () => jsonResponse({ ...HEALTH, admin_enabled: false }),
    });
    renderApp("/admin");

    expect(await screen.findByText(/Admin access is turned off/)).toBeInTheDocument();
    expect(screen.queryByLabelText("Password")).not.toBeInTheDocument();
  });

  it("returns to sign in when the session expires", async () => {
    mockApi({
      "GET /admin/session": () => jsonResponse({ authenticated: true }),
      "GET /admin/documents": unauthorized,
      "GET /admin/upload-limit": unauthorized,
    });
    renderApp("/admin");

    expect(await screen.findByLabelText("Password")).toBeInTheDocument();
  });

  it("signs out", async () => {
    const fetchMock = loggedIn([], { "POST /admin/logout": () => new Response(null, { status: 204 }) });
    renderApp("/admin");
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /sign out/i }));

    expect(await screen.findByLabelText("Password")).toBeInTheDocument();
    expect(callsTo(fetchMock, "POST", "/admin/logout")).toHaveLength(1);
  });
});

describe("Documents", () => {
  it("lists documents with details", async () => {
    loggedIn();
    renderApp("/admin");

    expect(await screen.findByText("Leave_Policy.txt")).toBeInTheDocument();
    expect(screen.getByText(/3 chunks/)).toBeInTheDocument();
    expect(screen.getByText("1 document(s)")).toBeInTheDocument();
  });

  it("shows an empty state", async () => {
    loggedIn([]);
    renderApp("/admin");
    expect(await screen.findByText(/No documents yet/)).toBeInTheDocument();
  });

  it("uploads files one by one and reports each result", async () => {
    const documents = [DOC];
    const fetchMock = loggedIn(documents, {
      "POST /admin/documents": (_url: string, init: RequestInit) => {
        const file = (init.body as FormData).get("file") as File;
        if (file.name === "copy.txt") {
          return jsonResponse({
            ...DOC,
            status: "duplicate",
            message: "'Leave_Policy.txt' is already in the knowledge base; skipped.",
          });
        }
        documents.push({ ...DOC, doc_id: "b".repeat(64), source: file.name });
        return jsonResponse(
          { ...DOC, source: file.name, status: "ingested", message: `Indexed '${file.name}' (2 chunks).` },
          201,
        );
      },
    });
    renderApp("/admin");
    const user = userEvent.setup({ applyAccept: false });

    await screen.findByText("Leave_Policy.txt");
    await user.upload(screen.getByTestId("file-input"), [
      new File(["remote work"], "Remote.docx"),
      new File(["same"], "copy.txt"),
      new File(["MZ"], "tool.exe"),
    ]);

    const uploads = within(screen.getByRole("list", { name: "Uploads" }));
    expect(await uploads.findByText("Indexed 'Remote.docx' (2 chunks).")).toBeInTheDocument();
    expect(await uploads.findByText(/already in the knowledge base/)).toBeInTheDocument();
    expect(await uploads.findByText(/Not a supported file type/)).toBeInTheDocument();

    // The unsupported file never reached the server; the list refreshed.
    expect(callsTo(fetchMock, "POST", "/admin/documents")).toHaveLength(2);
    expect(await screen.findByText("Remote.docx", { selector: "p.truncate.text-sm" })).toBeInTheDocument();
  });

  it("shows server-side upload errors", async () => {
    loggedIn([], {
      "POST /admin/documents": () =>
        jsonResponse({ detail: "'huge.pdf' is larger than the 20 MB limit." }, 413),
    });
    renderApp("/admin");
    const user = userEvent.setup();

    await screen.findByText(/No documents yet/);
    await user.upload(screen.getByTestId("file-input"), new File(["x"], "huge.pdf"));

    expect(await screen.findByText("'huge.pdf' is larger than the 20 MB limit.")).toBeInTheDocument();
  });

  it("deletes a document after confirmation", async () => {
    const documents = [DOC];
    const fetchMock = loggedIn(documents, {
      [`DELETE /admin/documents/${DOC.doc_id}`]: () => {
        documents.pop();
        return new Response(null, { status: 204 });
      },
    });
    renderApp("/admin");
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: "Delete Leave_Policy.txt" }));
    expect(screen.getByText("Delete this document?")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Cancel" }));
    expect(callsTo(fetchMock, "DELETE", `/admin/documents/${DOC.doc_id}`)).toHaveLength(0);

    await user.click(screen.getByRole("button", { name: "Delete Leave_Policy.txt" }));
    await user.click(screen.getByRole("button", { name: "Delete" }));

    expect(await screen.findByText(/No documents yet/)).toBeInTheDocument();
    expect(callsTo(fetchMock, "DELETE", `/admin/documents/${DOC.doc_id}`)).toHaveLength(1);
  });
});

describe("Upload size limit", () => {
  it("shows the current limit and changes it", async () => {
    let limit = LIMIT;
    const fetchMock = loggedIn([], {
      "GET /admin/upload-limit": () => jsonResponse(limit),
      "PUT /admin/upload-limit": (_url: string, init: RequestInit) => {
        limit = { ...LIMIT, max_upload_size_mb: JSON.parse(init.body as string).max_upload_size_mb };
        return jsonResponse(limit);
      },
    });
    renderApp("/admin");
    const user = userEvent.setup();

    expect(await screen.findByText("20 MB")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Change" }));
    const input = screen.getByLabelText("Max file size (MB)");
    expect(screen.getByText(/Between 1 and 200 MB/)).toBeInTheDocument();
    await user.clear(input);
    await user.type(input, "50");
    await user.click(screen.getByRole("button", { name: "Save" }));

    expect(await screen.findByText("50 MB")).toBeInTheDocument();
    expect(screen.getByText(/Applies to the next upload/)).toBeInTheDocument();
    expect(callsTo(fetchMock, "PUT", "/admin/upload-limit")).toEqual([{ max_upload_size_mb: 50 }]);
  });

  it("only allows whole numbers within the range", async () => {
    loggedIn([]);
    renderApp("/admin");
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: "Change" }));
    const input = screen.getByLabelText("Max file size (MB)");
    const save = screen.getByRole("button", { name: "Save" });

    for (const value of ["0", "201", "2.5"]) {
      await user.clear(input);
      await user.type(input, value);
      expect(save).toBeDisabled();
    }
    await user.clear(input);
    expect(save).toBeDisabled();
    await user.type(input, "200");
    expect(save).toBeEnabled();
  });

  it("cancel keeps the old limit", async () => {
    const fetchMock = loggedIn([]);
    renderApp("/admin");
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: "Change" }));
    await user.type(screen.getByLabelText("Max file size (MB)"), "5");
    await user.click(screen.getByRole("button", { name: "Cancel" }));

    expect(screen.getByText("20 MB")).toBeInTheDocument();
    expect(screen.queryByText(/Applies to the next upload/)).not.toBeInTheDocument();
    expect(callsTo(fetchMock, "PUT", "/admin/upload-limit")).toHaveLength(0);
  });

  it("shows why saving failed", async () => {
    loggedIn([], {
      "PUT /admin/upload-limit": () =>
        jsonResponse({ detail: "The upload limit must be between 1 and 200 MB." }, 422),
    });
    renderApp("/admin");
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: "Change" }));
    await user.click(screen.getByRole("button", { name: "Save" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("between 1 and 200 MB");
  });

  it("does not upload files over the limit", async () => {
    const fetchMock = loggedIn([], {
      "GET /admin/upload-limit": () => jsonResponse({ ...LIMIT, max_upload_size_mb: 1 }),
    });
    renderApp("/admin");
    const user = userEvent.setup();

    await screen.findByText("1 MB");
    const big = new File(["x"], "big.pdf");
    Object.defineProperty(big, "size", { value: 1024 * 1024 + 1 });
    await user.upload(screen.getByTestId("file-input"), big);

    expect(await screen.findByText("Larger than the 1 MB limit.")).toBeInTheDocument();
    expect(callsTo(fetchMock, "POST", "/admin/documents")).toHaveLength(0);
  });
});

describe("Instructions", () => {
  it("edits and saves the instructions", async () => {
    const fetchMock = loggedIn([], {
      "PUT /admin/instructions": (_url: string, init: RequestInit) =>
        jsonResponse({
          text: JSON.parse(init.body as string).text,
          updated_at: "2026-09-23T09:00:00+00:00",
        }),
    });
    renderApp("/admin");
    const user = userEvent.setup();

    await user.click(await screen.findByRole("tab", { name: /instructions/i }));
    const editor = await screen.findByLabelText("Agent instructions");
    expect(editor).toHaveValue("Be concise.");
    const save = screen.getByRole("button", { name: "Save instructions" });
    expect(save).toBeDisabled();

    await user.clear(editor);
    await user.type(editor, "Use bullet points.");
    expect(screen.getByText("Unsaved changes")).toBeInTheDocument();
    await user.click(save);

    expect(await screen.findByText(/Saved. The next answer/)).toBeInTheDocument();
    expect(callsTo(fetchMock, "PUT", "/admin/instructions")).toEqual([{ text: "Use bullet points." }]);
    await waitFor(() => expect(screen.queryByText("Unsaved changes")).not.toBeInTheDocument());
  });

  it("can discard unsaved changes", async () => {
    loggedIn();
    renderApp("/admin");
    const user = userEvent.setup();

    await user.click(await screen.findByRole("tab", { name: /instructions/i }));
    const editor = await screen.findByLabelText("Agent instructions");
    await user.type(editor, " Extra.");
    await user.click(screen.getByRole("button", { name: "Discard" }));

    expect(editor).toHaveValue("Be concise.");
  });
});

const MODELS = {
  current: "openai/gpt-oss-120b",
  options: [
    { id: "openai/gpt-oss-120b", label: "GPT-OSS 120B", description: "Best answer quality." },
    { id: "openai/gpt-oss-20b", label: "GPT-OSS 20B", description: "Fastest." },
    { id: "qwen/qwen3.8-27b", label: "Qwen 3.8 27B", description: "Strong reasoning." },
  ],
};

describe("Model", () => {
  it("shows the models with the current one selected, and switches", async () => {
    let current = MODELS.current;
    const fetchMock = loggedIn([], {
      "GET /admin/model": () => jsonResponse({ ...MODELS, current }),
      "PUT /admin/model": (_url: string, init: RequestInit) => {
        current = JSON.parse(init.body as string).model;
        return jsonResponse({ ...MODELS, current });
      },
    });
    renderApp("/admin");
    const user = userEvent.setup();

    await user.click(await screen.findByRole("tab", { name: /model/i }));
    const group = await screen.findByRole("radiogroup", { name: "Chat model" });
    const radios = within(group).getAllByRole("radio");
    expect(radios).toHaveLength(3);
    expect(within(group).getByRole("radio", { name: /GPT-OSS 120B/ })).toHaveAttribute("aria-checked", "true");

    await user.click(within(group).getByRole("radio", { name: /Qwen 3.8 27B/ }));

    expect(await screen.findByText("Now answering with Qwen 3.8 27B.")).toBeInTheDocument();
    expect(within(group).getByRole("radio", { name: /Qwen 3.8 27B/ })).toHaveAttribute("aria-checked", "true");
    expect(callsTo(fetchMock, "PUT", "/admin/model")).toEqual([{ model: "qwen/qwen3.8-27b" }]);
  });

  it("clicking the current model does nothing", async () => {
    const fetchMock = loggedIn([], { "GET /admin/model": () => jsonResponse(MODELS) });
    renderApp("/admin");
    const user = userEvent.setup();

    await user.click(await screen.findByRole("tab", { name: /model/i }));
    await user.click(await screen.findByRole("radio", { name: /GPT-OSS 120B/ }));

    expect(callsTo(fetchMock, "PUT", "/admin/model")).toHaveLength(0);
  });

  it("shows why a switch failed", async () => {
    loggedIn([], {
      "GET /admin/model": () => jsonResponse(MODELS),
      "PUT /admin/model": () =>
        jsonResponse({ detail: "'x' is not one of the available models." }, 422),
    });
    renderApp("/admin");
    const user = userEvent.setup();

    await user.click(await screen.findByRole("tab", { name: /model/i }));
    await user.click(await screen.findByRole("radio", { name: /GPT-OSS 20B/ }));

    expect(await screen.findByRole("alert")).toHaveTextContent("not one of the available models");
  });
});

describe("Re-index", () => {
  it("re-indexes saved files and reports the result", async () => {
    const documents: (typeof DOC)[] = [];
    const fetchMock = loggedIn(documents, {
      "POST /admin/documents/reindex": () => {
        documents.push(DOC);
        return jsonResponse({ indexed: 1, skipped: 2, failed: { "notes.xyz": "Not supported." } });
      },
    });
    renderApp("/admin");
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /re-index saved files/i }));

    expect(await screen.findByText(/1 file indexed, 2 already indexed, 1 could not be read/)).toBeInTheDocument();
    expect(screen.getByText("notes.xyz: Not supported.")).toBeInTheDocument();
    expect(await screen.findByText("Leave_Policy.txt")).toBeInTheDocument();
    expect(callsTo(fetchMock, "POST", "/admin/documents/reindex")).toHaveLength(1);
  });
});
