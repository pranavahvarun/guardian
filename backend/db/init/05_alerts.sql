-- Alerts from fusion (Stage 7.5). Safe to run more than once.
-- New database: runs automatically. Existing database: run it once with
--   docker exec -i guardian-db psql -U guardian -d guardian < backend/db/init/05_alerts.sql
--
-- One row per alert episode: from the first check-in or alert until it is
-- cleared, cancelled, or a caregiver resolves it. stage follows the alert
-- manager (core/alerts.py); status is what caregivers act on.
--   stage:   check_in / alert / caregiver / urgent / cleared /
--            cancelled by wearer / wearer answered OK
--   history: every event of the episode with its evidence
ALTER TABLE alerts ADD COLUMN IF NOT EXISTS stage      text NOT NULL DEFAULT 'alert';
ALTER TABLE alerts ADD COLUMN IF NOT EXISTS history    jsonb NOT NULL DEFAULT '[]';
ALTER TABLE alerts ADD COLUMN IF NOT EXISTS updated_at timestamptz NOT NULL DEFAULT now();
CREATE INDEX IF NOT EXISTS alerts_run_idx ON alerts (run_id);
