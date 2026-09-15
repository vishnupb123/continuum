import type { ReactNode } from "react";
import "./globals.css";

export const metadata = {
  title: "Continuum M0",
  description: "Longitudinal personal intelligence — M0 skeleton",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
