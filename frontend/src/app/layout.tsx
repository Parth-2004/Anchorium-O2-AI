/**
 * ==========================================================================
 * Root Layout — Anchorium Omni-Engine
 * ==========================================================================
 *
 * Sets up the global HTML structure, fonts, and metadata.
 * Uses Geist font family for a clean, modern developer aesthetic.
 */

import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Anchorium Omni-Engine | Zero-Knowledge Financial Intelligence",
  description:
    "Securely process financial documents using in-browser AI. " +
    "Your data never leaves your device. Built for foreign startup founders " +
    "expanding to India.",
  keywords: [
    "zero-knowledge",
    "financial AI",
    "GAAP to IndAS",
    "startup India",
    "corporate credit",
    "local RAG",
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
