import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PAB Operator Dashboard",
  description: "GovTech emergency-response operator dashboard"
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
