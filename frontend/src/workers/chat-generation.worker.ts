import { pipeline, TextGenerationPipeline, TextStreamer, ProgressInfo } from "@huggingface/transformers";

let generatorPipeline: TextGenerationPipeline | null = null;

// Xenova/Qwen1.5-0.5B-Chat is highly optimized for fast browser inference while remaining accurate
const MODEL_NAME = "Xenova/Qwen1.5-0.5B-Chat";

async function getGeneratorPipeline(): Promise<TextGenerationPipeline> {
  if (generatorPipeline) return generatorPipeline;

  self.postMessage({
    type: "status",
    message: "Downloading local LLM weights (~300MB, one-time only)...",
  });

  generatorPipeline = (await pipeline(
    "text-generation",
    MODEL_NAME,
    {
      progress_callback: (progress: ProgressInfo) => {
        if (progress.status === "done") {
          self.postMessage({
            type: "status",
            message: `Loaded chunk: ${progress.file ?? ""}`,
          });
        }
      },
    }
  )) as TextGenerationPipeline;

  self.postMessage({ type: "status", message: "✓ Local LLM engine ready." });
  return generatorPipeline;
}

self.addEventListener("message", async (event: MessageEvent) => {
  const { type, messages, messageId } = event.data;

  if (type !== "generate") return;

  try {
    const generator = await getGeneratorPipeline();

    // Apply the chat template using the tokenizer
    const prompt = generator.tokenizer.apply_chat_template(messages, {
      tokenize: false,
      add_generation_prompt: true,
    }) as string;

    const streamer = new TextStreamer(generator.tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: (text: string) => {
        self.postMessage({
          type: "stream-chunk",
          messageId,
          text,
        });
      }
    });

    await generator(prompt, {
      max_new_tokens: 512,
      temperature: 0.1, // Low temperature for high accuracy/factual answers
      repetition_penalty: 1.1,
      do_sample: false, // greedy decoding for factual extraction
      streamer,
    });

    self.postMessage({ type: "stream-complete", messageId });

  } catch (error) {
    self.postMessage({
      type: "error",
      messageId,
      message: `Local LLM generation failed: ${error instanceof Error ? error.message : String(error)}`,
    });
  }
});
