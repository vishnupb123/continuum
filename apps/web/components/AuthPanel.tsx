"use client";

import { FormEvent, useEffect, useState } from "react";
import JournalWorkspace from "./JournalWorkspace";

import {
  getCurrentUser,
  login,
  logout,
  register,
  type User,
} from "../lib/api";

type Mode = "login" | "register";

export default function AuthPanel() {
  const [user, setUser] = useState<User | null>(null);
  const [mode, setMode] = useState<Mode>("login");

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");

  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getCurrentUser()
      .then(setUser)
      .catch(() => setUser(null))
      .finally(() => setLoading(false));
  }, []);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    setSubmitting(true);
    setError(null);

    try {
      if (mode === "register") {
        await register({
          email,
          password,
          display_name: displayName || undefined,
        });
      }

      await login({
        email,
        password,
      });

      const authenticatedUser = await getCurrentUser();

      setUser(authenticatedUser);
      setPassword("");
      setEmail("");
      setDisplayName("");
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Authentication failed"
      );
    } finally {
      setSubmitting(false);
    }
  }

  async function handleLogout() {
    setError(null);

    try {
      await logout();
      setUser(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Logout failed");
    }
  }

  if (loading) {
    return <main>Checking your session...</main>;
  }

  if (user) {
  return (
    <JournalWorkspace
      user={user}
      onLogout={handleLogout}
    />
  );
}

  return (
    <main>
      <h1>Continuum</h1>

      <p>
        Understand how you change over time.
      </p>

      <div>
        <button
          type="button"
          onClick={() => {
            setMode("login");
            setError(null);
          }}
        >
          Log in
        </button>

        <button
          type="button"
          onClick={() => {
            setMode("register");
            setError(null);
          }}
        >
          Create account
        </button>
      </div>

      <form onSubmit={handleSubmit}>
        {mode === "register" && (
          <div>
            <label htmlFor="displayName">Name</label>

            <input
              id="displayName"
              value={displayName}
              onChange={(event) => setDisplayName(event.target.value)}
              maxLength={100}
            />
          </div>
        )}

        <div>
          <label htmlFor="email">Email</label>

          <input
            id="email"
            type="email"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            required
          />
        </div>

        <div>
          <label htmlFor="password">Password</label>

          <input
            id="password"
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            minLength={mode === "register" ? 8 : 1}
            required
          />
        </div>

        {error && <p role="alert">{error}</p>}

        <button type="submit" disabled={submitting}>
          {submitting
            ? "Please wait..."
            : mode === "login"
              ? "Log in"
              : "Create account"}
        </button>
      </form>
    </main>
  );
}