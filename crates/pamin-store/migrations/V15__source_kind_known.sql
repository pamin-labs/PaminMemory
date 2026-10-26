-- `sources.kind` was left open while `manual` was the only kind anything
-- wrote, because a constraint listing kinds no ingest path produced would have
-- been a rule nothing enforced. `pamin ingest` writes the second, `file`, so
-- the column now admits exactly the kinds a variant produces, as every other
-- labelled column does.
ALTER TABLE sources
    ADD CONSTRAINT sources_kind_known CHECK (kind IN ('manual', 'file'));
