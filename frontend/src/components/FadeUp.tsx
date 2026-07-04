"use client";
import { motion } from 'framer-motion';
import {
  CSSProperties,
  ComponentType,
  MouseEventHandler,
  ReactNode,
} from 'react';

type FadeUpProps = {
  children: ReactNode;
  delay?: number;
  duration?: number;
  y?: number;
  className?: string;
  style?: CSSProperties;
  as?: 'div' | 'section' | 'span' | 'h1' | 'h2' | 'h3' | 'p' | 'nav' | 'button';
  once?: boolean;
  onClick?: () => void;
  onMouseEnter?: MouseEventHandler<HTMLElement>;
  onMouseLeave?: MouseEventHandler<HTMLElement>;
};

type FadeUpMotionProps = {
  children: ReactNode;
  className?: string;
  style?: CSSProperties;
  initial: { opacity: number; y: number };
  whileInView: { opacity: number; y: number };
  viewport: { once: boolean; amount: number };
  transition: {
    duration: number;
    delay: number;
    ease: number[];
  };
  onClick?: () => void;
  onMouseEnter?: MouseEventHandler<HTMLElement>;
  onMouseLeave?: MouseEventHandler<HTMLElement>;
};

export function FadeUp({
  children, delay = 0, duration = 0.7, y = 24,
  className, style, as = 'div', once = true, onClick, onMouseEnter, onMouseLeave
}: FadeUpProps) {
  const Tag = motion[as as keyof typeof motion] as ComponentType<FadeUpMotionProps>;
  return (
    <Tag
      className={className}
      style={style}
      initial={{ opacity: 0, y }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once, amount: 0.2 }}
      transition={{ duration, delay, ease: [0.22, 1, 0.36, 1] }}
      onClick={onClick}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
    >
      {children}
    </Tag>
  );
}
