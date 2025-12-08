/**
 * Application-wide constants
 */

export const PROGRESS = {
  /** Maximum simulated progress before actual completion */
  MAX_SIMULATED: 50,
  /** Progress just before API call completes */
  BEFORE_COMPLETE: 95,
  /** Final progress value */
  COMPLETE: 100,
} as const;

export const PROGRESS_UPDATE = {
  /** Interval in milliseconds for progress updates */
  INTERVAL_MS: 200,
  /** Random increment range for progress simulation */
  INCREMENT_MIN: 1,
  INCREMENT_MAX: 3,
  /** Delay in milliseconds before resetting progress */
  RESET_DELAY_MS: 500,
} as const;

export const UI = {
  /** Delay in milliseconds for image scale calculation */
  SCALE_UPDATE_DELAY_MS: 100,
} as const;

