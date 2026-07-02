import fs from 'fs';
import path from 'path';
import { pipeline } from '@huggingface/transformers';
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const pdfParse = require('pdf-parse');

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

// Absolute or relative path to the PDF corpus
const CORPUS_DIR = path.resolve('../backend/Verified Indian Regulatory Document Corpus/A. RBI Master Directions-Foreeign Exchange Management');

// Output file in the frontend's public directory
const OUTPUT_FILE = path.resolve('public/regulatory_embeddings.json');

// Chunking Configuration
const CHUNK_SIZE = 500;
const OVERLAP = 50;

// ---------------------------------------------------------------------------
// Text Chunking Function (matching the frontend's logic)
// ---------------------------------------------------------------------------
function chunkText(text, chunkSize = CHUNK_SIZE, overlap = OVERLAP) {
  const chunks = [];
  const cleanedText = text.replace(/\s+/g, " ").trim();

  if (cleanedText.length <= chunkSize) {
    return [cleanedText];
  }

  let start = 0;
  while (start < cleanedText.length) {
    const end = Math.min(start + chunkSize, cleanedText.length);
    const chunk = cleanedText.slice(start, end).trim();

    if (chunk.length > 0) {
      chunks.push(chunk);
    }

    start += chunkSize - overlap;
  }

  return chunks;
}

// ---------------------------------------------------------------------------
// Main Pipeline
// ---------------------------------------------------------------------------
async function main() {
  console.log(`[1/5] Starting Regulatory Embeddings Pre-computation...`);
  console.log(`Corpus Directory: ${CORPUS_DIR}`);

  if (!fs.existsSync(CORPUS_DIR)) {
    console.error(`ERROR: Corpus directory not found at ${CORPUS_DIR}`);
    process.exit(1);
  }

  const files = fs.readdirSync(CORPUS_DIR).filter(file => file.endsWith('.pdf'));
  console.log(`Found ${files.length} PDF documents.`);

  // Load the model
  console.log(`[2/5] Loading Transformers.js Model (Xenova/all-MiniLM-L6-v2)...`);
  const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

  const allEmbeddings = [];

  for (let i = 0; i < files.length; i++) {
    const fileName = files[i];
    const filePath = path.join(CORPUS_DIR, fileName);
    
    console.log(`\n[3/5] Processing file ${i + 1}/${files.length}: ${fileName}`);
    
    try {
      // 1. Read PDF
      const dataBuffer = fs.readFileSync(filePath);
      const pdfData = await pdfParse(dataBuffer);
      const text = pdfData.text;

      if (!text || text.trim() === '') {
        console.warn(`Warning: No extractable text found in ${fileName}`);
        continue;
      }

      // 2. Chunk Text
      const chunks = chunkText(text);
      console.log(`      Extracted text and split into ${chunks.length} chunks.`);

      // 3. Generate Embeddings
      for (let j = 0; j < chunks.length; j++) {
        const chunkTextContent = chunks[j];
        
        // Generate embedding (returns a tensor)
        const output = await extractor(chunkTextContent, { pooling: 'mean', normalize: true });
        
        // Convert Float32Array to standard array
        const vector = Array.from(output.data);

        allEmbeddings.push({
          text: chunkTextContent,
          vector: vector,
          documentName: fileName
        });
      }
      
      console.log(`      Successfully embedded ${chunks.length} chunks for ${fileName}.`);
      
    } catch (error) {
      console.error(`      ERROR processing ${fileName}:`, error);
    }
  }

  console.log(`\n[4/5] Pre-computation complete. Total embeddings generated: ${allEmbeddings.length}`);

  // Save to JSON
  console.log(`[5/5] Saving to ${OUTPUT_FILE}...`);
  fs.mkdirSync(path.dirname(OUTPUT_FILE), { recursive: true });
  fs.writeFileSync(OUTPUT_FILE, JSON.stringify(allEmbeddings));
  
  console.log(`\n✅ Success! Saved ${allEmbeddings.length} chunks to public/regulatory_embeddings.json`);
}

main().catch(console.error);
