import React from "react";
import { Mic, Camera, Settings } from "lucide-react";
import { cn } from "@/lib/utils";

interface HeaderProps {
  className?: string;
}

const Header: React.FC<HeaderProps> = ({ className }) => {
  return (
    <header
      className={cn(
        "w-full py-4 px-6 flex items-center justify-between bg-white/50 backdrop-blur-md border-b border-slate-200/70 z-10",
        className
      )}
    >
      <div className="flex items-center space-x-2">
        <div className="h-8 w-8 rounded-full bg-gradient-to-tr from-blue-500 to-indigo-600 flex items-center justify-center">
          <Camera className="h-4 w-4 text-white" />
        </div>
        <div className="flex flex-col">
          <h1 className="text-lg font-medium leading-none tracking-tight">
            MIROB
          </h1>
          <p className="text-xs text-muted-foreground mt-0.5">
            Conversational Service Robot
          </p>
        </div>
      </div>

      <div className="flex items-center space-x-2">
        <div className="text-xs px-2.5 py-0.5 rounded-full bg-blue-100 text-blue-800 font-medium animate-pulse-gentle">
          UOM
        </div>
        <button className="p-2 rounded-full hover:bg-slate-100 transition-colors">
          <Settings className="h-4 w-4 text-slate-600" />
        </button>
      </div>
    </header>
  );
};

export default Header;
