import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Machine Learning — Regression Interactive Demo",
  description:
    "An interactive laboratory for exploring linear regression, noise, and outliers.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
