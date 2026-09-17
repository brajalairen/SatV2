/** Shared primitives. Deliberately few: one button, one floating surface, one tooltip, one field.
 *  Everything else composes from these so the interface stays visually consistent. */

import {
  cloneElement,
  createContext,
  isValidElement,
  useContext,
  useEffect,
  useId,
  useRef,
  useState,
  type ButtonHTMLAttributes,
  type ReactNode,
} from "react";

export function cx(...parts: (string | false | null | undefined)[]): string {
  return parts.filter(Boolean).join(" ");
}

/* ------------------------------------------------------------------ surfaces */

/** The one elevation treatment. Everything that floats over the map uses it. */
export function Surface({
  children,
  className,
  raised,
  ...rest
}: { children: ReactNode; className?: string; raised?: boolean } & React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      {...rest}
      className={cx(
        "rounded-[var(--radius-card)] border border-line bg-surface",
        raised ? "shadow-[var(--shadow-lg)]" : "shadow-[var(--shadow-md)]",
        className,
      )}
    >
      {children}
    </div>
  );
}

/* ------------------------------------------------------------------ tooltip */

/**
 * Tooltip for icon-only controls. The label is also applied as `aria-label` on the child by the
 * caller, so screen readers never depend on hover.
 */
export function Tooltip({
  label,
  side = "right",
  children,
}: {
  label: string;
  side?: "right" | "top" | "left" | "bottom";
  children: ReactNode;
}) {
  const [open, setOpen] = useState(false);
  const position = {
    right: "left-full top-1/2 -translate-y-1/2 ml-2",
    left: "right-full top-1/2 -translate-y-1/2 mr-2",
    top: "bottom-full left-1/2 -translate-x-1/2 mb-2",
    bottom: "top-full left-1/2 -translate-x-1/2 mt-2",
  }[side];

  return (
    <span
      className="relative inline-flex"
      onPointerEnter={() => setOpen(true)}
      onPointerLeave={() => setOpen(false)}
      onFocusCapture={() => setOpen(true)}
      onBlurCapture={() => setOpen(false)}
    >
      {children}
      {open && (
        <span
          role="tooltip"
          className={cx(
            "pointer-events-none absolute z-50 whitespace-nowrap rounded-md bg-ink px-2 py-1",
            "text-[11px] font-medium text-surface shadow-[var(--shadow-md)]",
            position,
          )}
        >
          {label}
        </span>
      )}
    </span>
  );
}

/* ------------------------------------------------------------------ buttons */

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "quiet" | "outline";
  size?: "sm" | "md";
};

export function Button({ variant = "quiet", size = "md", className, ...rest }: ButtonProps) {
  const variants = {
    primary: "bg-accent text-accent-ink hover:brightness-110",
    quiet: "text-ink hover:bg-hover",
    outline: "border border-line text-ink hover:bg-hover hover:border-line-strong",
  }[variant];

  return (
    <button
      type="button"
      {...rest}
      className={cx(
        "inline-flex items-center justify-center gap-1.5 rounded-[var(--radius-sm)] font-medium",
        "transition-colors disabled:cursor-not-allowed disabled:opacity-45",
        size === "sm" ? "h-7 px-2 text-[12px]" : "h-9 px-3 text-[13px]",
        variants,
        className,
      )}
    />
  );
}

/**
 * Square icon button. `label` is both the tooltip and the accessible name, so an icon never has to
 * carry meaning on its own.
 */
export function IconButton({
  label,
  side = "right",
  active,
  className,
  children,
  showTooltip = true,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  label: string;
  side?: "right" | "top" | "left" | "bottom";
  active?: boolean;
  showTooltip?: boolean;
}) {
  const button = (
    <button
      type="button"
      aria-label={label}
      aria-pressed={active}
      {...rest}
      className={cx(
        "inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-[var(--radius-sm)]",
        "transition-colors disabled:cursor-not-allowed disabled:opacity-40",
        active ? "bg-accent-soft text-accent" : "text-muted hover:bg-hover hover:text-ink",
        className,
      )}
    >
      {children}
    </button>
  );
  return showTooltip ? (
    <Tooltip label={label} side={side}>
      {button}
    </Tooltip>
  ) : (
    button
  );
}

/* ------------------------------------------------------------------ fields */

/** Labelled control. The generated id is passed to a single element child so the label stays wired
 *  to the real input; anything else (a group of buttons) is rendered inside the label instead. */
export function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  const id = useId();
  const control = isValidElement<{ id?: string }>(children) ? cloneElement(children, { id }) : children;
  return (
    <label htmlFor={id} className="block">
      <span className="mb-1 block text-[11px] font-medium tracking-wide text-muted uppercase">{label}</span>
      {control}
      {hint && <span className="mt-1 block text-[11px] text-faint">{hint}</span>}
    </label>
  );
}

export const inputClass =
  "w-full rounded-[var(--radius-sm)] border border-line bg-sunken px-2.5 py-1.5 text-[13px] text-ink " +
  "placeholder:text-faint transition-colors hover:border-line-strong focus:border-accent focus:outline-none " +
  "disabled:cursor-not-allowed disabled:opacity-50";

/** Small segmented control for two or three mutually exclusive choices. */
export function Segmented<T extends string>({
  value,
  options,
  onChange,
  disabled,
}: {
  value: T;
  options: { value: T; label: string }[];
  onChange: (value: T) => void;
  disabled?: boolean;
}) {
  return (
    <div role="group" className="inline-flex rounded-[var(--radius-sm)] border border-line bg-sunken p-0.5">
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          disabled={disabled}
          aria-pressed={value === option.value}
          onClick={() => onChange(option.value)}
          className={cx(
            "rounded-[6px] px-2.5 py-1 text-[12px] font-medium transition-colors disabled:opacity-50",
            value === option.value ? "bg-surface text-ink shadow-[var(--shadow-sm)]" : "text-muted hover:text-ink",
          )}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ disclosure */

/** Collapsible section. Advanced controls live inside these so the panel opens quiet. */
export function SectionGroup({
  title,
  note,
  defaultOpen = false,
  disabled,
  children,
}: {
  title: string;
  note?: string;
  defaultOpen?: boolean;
  disabled?: boolean;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section className="border-b border-line last:border-b-0">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        className="flex w-full items-center justify-between gap-2 px-4 py-3 text-left transition-colors hover:bg-hover"
      >
        <span className="flex flex-col">
          <span className={cx("text-[13px] font-semibold", disabled ? "text-muted" : "text-ink")}>{title}</span>
          {note && <span className="mt-0.5 text-[11px] leading-snug text-faint">{note}</span>}
        </span>
        <Chevron open={open} />
      </button>
      {open && <div className="px-4 pt-0.5 pb-4">{children}</div>}
    </section>
  );
}

function Chevron({ open }: { open: boolean }) {
  return (
    <svg
      viewBox="0 0 16 16"
      aria-hidden="true"
      className={cx("h-3.5 w-3.5 shrink-0 text-faint transition-transform", open && "rotate-180")}
    >
      <path d="M4 6l4 4 4-4" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  );
}

/* ------------------------------------------------------------------ misc */

export function Spinner({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true" className={cx("h-4 w-4 animate-spin", className)}>
      <circle cx="8" cy="8" r="6.5" fill="none" stroke="currentColor" strokeOpacity="0.25" strokeWidth="2" />
      <path d="M14.5 8A6.5 6.5 0 0 0 8 1.5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

/** Labels a capability that is genuinely not wired up, so it can never read as working. */
export function NotConnected({ children }: { children: ReactNode }) {
  return (
    <span className="inline-flex items-center rounded-full border border-line px-1.5 py-0.5 text-[10px] font-medium tracking-wide text-faint uppercase">
      {children}
    </span>
  );
}

/** Dismissable popover anchored to a trigger, used by the upload and overflow menus. */
export function Popover({
  open,
  onClose,
  children,
  className,
}: {
  open: boolean;
  onClose: () => void;
  children: ReactNode;
  className?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onPointerDown = (event: PointerEvent) => {
      if (!ref.current?.contains(event.target as Node)) onClose();
    };
    const onKeyDown = (event: KeyboardEvent) => event.key === "Escape" && onClose();
    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open, onClose]);

  if (!open) return null;
  return (
    <div ref={ref} className={cx("absolute z-40", className)}>
      <Surface raised className="overflow-hidden py-1">
        {children}
      </Surface>
    </div>
  );
}

export function MenuItem({
  icon,
  children,
  note,
  disabled,
  onClick,
}: {
  icon?: ReactNode;
  children: ReactNode;
  note?: string;
  disabled?: boolean;
  onClick?: () => void;
}) {
  const content = (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      className={cx(
        "flex w-full items-center gap-2.5 px-3 py-2 text-left text-[13px] transition-colors",
        disabled ? "cursor-not-allowed text-faint" : "text-ink hover:bg-hover",
      )}
    >
      {icon && <span className="shrink-0 text-muted">{icon}</span>}
      <span className="flex-1">{children}</span>
      {disabled && note && <NotConnected>{note}</NotConnected>}
    </button>
  );
  return disabled && note ? <Tooltip label={note} side="top">{content}</Tooltip> : content;
}

/* ------------------------------------------------------------------ toast */

const ToastContext = createContext<(message: string) => void>(() => {});
export const useToast = () => useContext(ToastContext);

export function ToastHost({ children }: { children: ReactNode }) {
  const [message, setMessage] = useState<string | null>(null);

  useEffect(() => {
    if (!message) return;
    const timer = setTimeout(() => setMessage(null), 4000);
    return () => clearTimeout(timer);
  }, [message]);

  return (
    <ToastContext.Provider value={setMessage}>
      {children}
      {message && (
        <div
          role="status"
          aria-live="polite"
          className="pointer-events-none fixed bottom-28 left-1/2 z-[60] -translate-x-1/2 px-4"
        >
          <Surface raised className="px-3 py-2 text-[13px] text-ink">
            {message}
          </Surface>
        </div>
      )}
    </ToastContext.Provider>
  );
}
