import React from 'react';

interface KpiCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  annotation?: string;
  badge?: {
    text: string;
    variant: 'neutral' | 'success' | 'warning' | 'info';
  };
  icon?: React.ReactNode;
}

export const KpiCard: React.FC<KpiCardProps> = ({
  title,
  value,
  subtitle,
  annotation,
  badge,
  icon,
}) => {
  const badgeClasses = {
    neutral: 'bg-slate-100 text-slate-700 border-slate-200',
    success: 'bg-emerald-50 text-emerald-700 border-emerald-200',
    warning: 'bg-amber-50 text-amber-800 border-amber-200',
    info: 'bg-blue-50 text-blue-700 border-blue-200',
  }[badge?.variant || 'neutral'];

  return (
    <div className="enterprise-card p-5 flex flex-col justify-between">
      <div className="flex items-start justify-between">
        <span className="text-xs font-semibold text-text-secondary uppercase tracking-wider">
          {title}
        </span>
        {icon && <div className="text-slate-400">{icon}</div>}
      </div>

      <div className="my-2">
        <div className="text-2xl font-bold text-text-primary tracking-tight tabular-nums">
          {value}
        </div>
        {subtitle && (
          <div className="text-xs text-text-muted mt-0.5 font-normal">
            {subtitle}
          </div>
        )}
      </div>

      {(annotation || badge) && (
        <div className="pt-2 border-t border-border-subtle/80 flex items-center justify-between text-[11px]">
          {annotation && <span className="text-text-muted italic">{annotation}</span>}
          {badge && (
            <span className={`inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-semibold border ${badgeClasses}`}>
              {badge.text}
            </span>
          )}
        </div>
      )}
    </div>
  );
};

