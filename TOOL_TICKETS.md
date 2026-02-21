# Dev Arena Tool Upgrade Tickets

This file tracks platform-engineer upgrade requests issued by the Senior Dev Coach when it identifies that a tool is physically inadequate or unintuitive for the Junior to use effectively.

## Open Tickets

### 1. `glob_search`
**Date Found:** Feb 21, 2026
**Issue:** Junior expects `glob_search` to find contents within files, but it only matches file paths.
**Recommendation:** Fix `glob_search` to return actual file path results when searching for a string pattern within a directory, or clearly document that it only matches filenames/paths (not file contents), so the Junior knows to use `grep` instead for content searches.
**Status:** ✅ RESOLVED (Updated tool description in `real_world_agent.py` to clarify it only matches paths and to redirect to `grep` for contents)

### 2. `file_patch`
**Date Found:** Feb 20, 2026
**Issue:** `file_patch` struggles with multi-line indentation and silently fails or corrupts files.
**Recommendation:** The `file_patch` tool is not properly detecting and applying changes when the exact string match spans multiple lines with specific indentation; the tool should validate that patches were successfully applied by re-reading the file and comparing before/after content.
**Status:** ✅ RESOLVED (Added a write-verification step to `file_patch` that reads back the file from disk and compares it to the intended new content to guarantee the patch was applied exactly as intended)

### New Ticket: `glob_search`
**Date Found:** Feb 21, 2026
**Recommendation:** Update the glob_search tool to accept and document an optional 'glob_filter' parameter (or validate/raise a clearer error), since users reasonably attempt to filter file types and the unexpected keyword error derails workflows.
**Status:** 🔴 OPEN
