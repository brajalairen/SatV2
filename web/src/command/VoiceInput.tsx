/** Dictation through the browser's own Web Speech API. Nothing is sent to our server, and where
 *  the browser does not support it the button is simply not rendered. */

import { useEffect, useRef, useState } from "react";
import { Mic, MicOff } from "lucide-react";
import { IconButton } from "../ui/primitives";

/** Minimal shape of the parts of SpeechRecognition this component touches. */
interface Recognition extends EventTarget {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  start(): void;
  stop(): void;
  onresult: ((event: SpeechRecognitionLikeEvent) => void) | null;
  onerror: ((event: { error: string }) => void) | null;
  onend: (() => void) | null;
}

interface SpeechRecognitionLikeEvent {
  results: ArrayLike<ArrayLike<{ transcript: string }> & { isFinal: boolean }>;
}

type RecognitionConstructor = new () => Recognition;

function getConstructor(): RecognitionConstructor | null {
  const scope = window as unknown as {
    SpeechRecognition?: RecognitionConstructor;
    webkitSpeechRecognition?: RecognitionConstructor;
  };
  return scope.SpeechRecognition ?? scope.webkitSpeechRecognition ?? null;
}

export function VoiceInput({
  onTranscript,
  disabled,
}: {
  onTranscript: (text: string) => void;
  disabled?: boolean;
}) {
  const [supported] = useState(() => getConstructor() !== null);
  const [listening, setListening] = useState(false);
  const [denied, setDenied] = useState(false);
  const recognition = useRef<Recognition | null>(null);

  useEffect(() => {
    return () => recognition.current?.stop();
  }, []);

  if (!supported) return null;

  const toggle = () => {
    if (listening) {
      recognition.current?.stop();
      return;
    }
    const Constructor = getConstructor();
    if (!Constructor) return;

    const instance = new Constructor();
    instance.lang = navigator.language || "en-US";
    instance.continuous = false;
    instance.interimResults = false;

    instance.onresult = (event) => {
      const parts: string[] = [];
      for (let i = 0; i < event.results.length; i += 1) {
        const result = event.results[i];
        if (result?.isFinal && result[0]) parts.push(result[0].transcript);
      }
      const text = parts.join(" ").trim();
      if (text) onTranscript(text);
    };
    instance.onerror = (event) => {
      if (event.error === "not-allowed" || event.error === "service-not-allowed") setDenied(true);
      setListening(false);
    };
    instance.onend = () => setListening(false);

    recognition.current = instance;
    setDenied(false);
    setListening(true);
    instance.start();
  };

  const label = denied
    ? "Microphone access was blocked"
    : listening
      ? "Stop dictating"
      : "Dictate a question";

  return (
    <IconButton
      label={label}
      side="top"
      disabled={disabled || denied}
      active={listening}
      onClick={toggle}
      className={listening ? "text-danger" : undefined}
    >
      {denied ? <MicOff className="h-[18px] w-[18px]" strokeWidth={1.75} /> : <Mic className="h-[18px] w-[18px]" strokeWidth={1.75} />}
    </IconButton>
  );
}
