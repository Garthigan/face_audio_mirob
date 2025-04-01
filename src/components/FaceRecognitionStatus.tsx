
import React from 'react';
import { Eye, Activity, Loader, CheckCircle } from 'lucide-react';
import { cn } from '@/lib/utils';
import { RecognitionStatus } from '@/utils/types';

interface FaceRecognitionStatusProps {
  className?: string;
  status: RecognitionStatus;
}

const FaceRecognitionStatus: React.FC<FaceRecognitionStatusProps> = ({ 
  className, 
  status 
}) => {
  return (
    <div className={cn(
      'glass-panel rounded-xl p-4 animate-fade-in',
      className
    )}>
      <h3 className="text-sm font-medium text-slate-700 mb-3 flex items-center">
        <Activity className="h-4 w-4 mr-1.5 text-blue-500" />
        System Status
      </h3>
      
      <div className="space-y-2">
        <StatusItem 
          icon={Eye} 
          label="Face Detection" 
          status={status.isDetecting ? 'active' : 'idle'} 
        />
        
        <StatusItem 
          icon={Loader} 
          label="Audio Processing" 
          status={status.isRecording ? 'recording' : (status.isProcessing ? 'processing' : 'idle')} 
        />
        
        <StatusItem 
          icon={CheckCircle} 
          label="Last Updated" 
          status="info"
          infoText={new Date(status.lastUpdated).toLocaleTimeString()}
        />
      </div>
    </div>
  );
};

interface StatusItemProps {
  icon: React.ElementType;
  label: string;
  status: 'active' | 'idle' | 'recording' | 'processing' | 'info';
  infoText?: string;
}

const StatusItem: React.FC<StatusItemProps> = ({ 
  icon: Icon, 
  label, 
  status, 
  infoText 
}) => {
  const getStatusColor = () => {
    switch (status) {
      case 'active':
        return 'bg-green-100 text-green-700';
      case 'recording':
        return 'bg-red-100 text-red-700';
      case 'processing':
        return 'bg-yellow-100 text-yellow-700';
      case 'idle':
        return 'bg-slate-100 text-slate-700';
      case 'info':
        return 'bg-blue-50 text-blue-700';
      default:
        return 'bg-slate-100 text-slate-700';
    }
  };

  const getStatusText = () => {
    switch (status) {
      case 'active':
        return 'Active';
      case 'recording':
        return 'Recording';
      case 'processing':
        return 'Processing';
      case 'idle':
        return 'Idle';
      case 'info':
        return infoText || '';
      default:
        return 'Unknown';
    }
  };

  return (
    <div className="flex items-center justify-between bg-white/60 backdrop-blur-sm rounded-md py-2 px-3 border border-slate-200/60">
      <div className="flex items-center">
        <Icon className="h-3.5 w-3.5 text-slate-600 mr-2" />
        <span className="text-xs text-slate-700">{label}</span>
      </div>
      <span className={`text-xs px-2 py-0.5 rounded-full ${getStatusColor()}`}>
        {getStatusText()}
      </span>
    </div>
  );
};

export default FaceRecognitionStatus;
