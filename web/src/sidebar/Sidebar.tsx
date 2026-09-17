/** Left sidebar. Collapsed it is an icon rail with the everyday actions; expanded it reveals the
 *  detailed controls. Nothing advanced is reachable until the user asks for it. */

import { CircleHelp, Layers, MapPin, Search, Bookmark, SlidersHorizontal, X } from "lucide-react";
import { useAppStore, type SidebarSection } from "../state/useAppStore";
import { cx, IconButton, Surface } from "../ui/primitives";
import { SearchPanel } from "./SearchPanel";
import { AreaSelectionTools } from "./AreaSelectionTools";
import { LayersPanel } from "./LayersPanel";
import { SavedAreas } from "./SavedAreas";
import { HelpPanel } from "./HelpPanel";

const ITEMS: { id: Exclude<SidebarSection, null>; label: string; icon: typeof Search }[] = [
  { id: "search", label: "Search location", icon: Search },
  { id: "select", label: "Select area", icon: MapPin },
  { id: "layers", label: "Images and imagery settings", icon: Layers },
  { id: "saved", label: "Saved areas", icon: Bookmark },
];

const TITLES: Record<Exclude<SidebarSection, null>, string> = {
  search: "Search location",
  select: "Select area",
  layers: "Images",
  saved: "Saved areas",
  help: "Help",
};

export function Sidebar() {
  const open = useAppStore((s) => s.sidebarOpen);
  const section = useAppStore((s) => s.section);
  const openSection = useAppStore((s) => s.openSection);
  const closeSidebar = useAppStore((s) => s.closeSidebar);
  const layerCount = useAppStore((s) => s.layers.length);

  return (
    <div className="pointer-events-none absolute top-4 bottom-4 left-4 z-30 flex items-start gap-3">
      {/* Collapsed rail: always present, the entry point to everything else. */}
      <Surface className="pointer-events-auto flex flex-col gap-1 p-1.5">
        {ITEMS.map((item) => (
          <div key={item.id} className="relative">
            <IconButton
              label={item.label}
              active={open && section === item.id}
              onClick={() => openSection(item.id)}
            >
              <item.icon className="h-[18px] w-[18px]" strokeWidth={1.75} />
            </IconButton>
            {item.id === "layers" && layerCount > 0 && (
              <span
                aria-hidden="true"
                className="pointer-events-none absolute top-1 right-1 flex h-4 min-w-4 items-center justify-center rounded-full bg-accent px-1 text-[10px] font-semibold text-accent-ink"
              >
                {layerCount}
              </span>
            )}
          </div>
        ))}
        <span className="mx-1.5 my-0.5 h-px bg-line" />
        <IconButton label="Help" active={open && section === "help"} onClick={() => openSection("help")}>
          <CircleHelp className="h-[18px] w-[18px]" strokeWidth={1.75} />
        </IconButton>
      </Surface>

      {/* Expanded panel: only mounted when a section is chosen. */}
      {open && section && (
        <Surface
          raised
          className={cx(
            "pointer-events-auto flex max-h-full w-[320px] flex-col overflow-hidden",
            "max-sm:fixed max-sm:inset-x-4 max-sm:top-4 max-sm:bottom-24 max-sm:w-auto",
          )}
        >
          <header className="flex shrink-0 items-center justify-between border-b border-line px-4 py-3">
            <h2 className="text-[13px] font-semibold text-ink">{TITLES[section]}</h2>
            <IconButton label="Close panel" side="left" onClick={closeSidebar} className="-mr-1.5 h-7 w-7">
              <X className="h-4 w-4" strokeWidth={2} />
            </IconButton>
          </header>
          <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain">
            {section === "search" && <SearchPanel />}
            {section === "select" && <AreaSelectionTools />}
            {section === "layers" && <LayersPanel />}
            {section === "saved" && <SavedAreas />}
            {section === "help" && <HelpPanel />}
          </div>
        </Surface>
      )}
    </div>
  );
}

/** Shared empty state, so every panel says what to do next in the same voice. */
export function EmptyState({ icon: Icon, children }: { icon: typeof SlidersHorizontal; children: React.ReactNode }) {
  return (
    <div className="flex flex-col items-center gap-2 px-6 py-10 text-center">
      <Icon className="h-5 w-5 text-faint" strokeWidth={1.5} />
      <p className="text-[12px] leading-relaxed text-muted">{children}</p>
    </div>
  );
}
