# Proposals: ticket -> proposal -> vote -> job -> review (design, 2026-11-15)

Status: design only. Nothing here is implemented. Written by claude at the
admin's request (messages #112, #117) for discussion under review.

## What exists today (checked 2026-11-15 against main 9630c59e)

- `issue_tickets.py`: account-owned incident tickets, statuses
  `open | investigating | resolved`, an append-only `issue_ticket_events`
  history with optimistic `version`, admin from `account_roles`.
- `tickets.py` (`/api/tickets`): legacy tickets.
- `branch_review.py` (`/api/reviews`): one pushed branch at one commit. Rule
  (`standing`): an admin decision settles it; otherwise silence is consent at
  `accept_after`; one standing peer objection stops the clock and waits for an
  admin. `branch_gate.py` lands what the rows allow.
- `/api/paper/proposals` is paper-trading strategies, unrelated.
- There is no feature-ticket kind, no proposal, no vote and no link from a
  ticket to the review that implements it.

## The flow asked for

    ticket (issue | feature)
      -> proposal (a written plan answering the ticket)
      -> vote period
      -> implementation job (someone claims it)
      -> change review(s)            (exists: branch_reviews)
      -> land                        (exists: branch_gate)
      -> ticket resolved

## Proposed shape: reuse the two machines already here

The design adds no new decision rule. It applies the review rule to a
proposal, and adds one link from tickets to reviews.

1. Ticket kind. Add `kind TEXT NOT NULL DEFAULT 'issue' CHECK(kind IN
   ('issue','feature'))` to `issue_tickets` (schema version 2). The Issues page
   gets a "Feature ticket" button beside "Issue ticket". Statuses gain
   `proposed`, `accepted`, `in_progress` between `open` and `resolved`;
   every move is still an `issue_ticket_events` row.

2. Proposal. A table `ticket_proposals(id, ticket_id, author_id, body,
   created_at, vote_until, status open|accepted|rejected|withdrawn, decided_by,
   decided_at, note)`. Any signed-in account may propose on a ticket it can
   see. A ticket may hold several proposals; at most one is accepted.

3. Vote. `ticket_proposal_votes(proposal_id, user_id, verdict agree|object,
   note, at, PRIMARY KEY(proposal_id,user_id))`: the same shape as
   `branch_review_remarks`, and the same rule as `standing`:
   - an admin decision settles it at once;
   - otherwise, with no standing objection, it is accepted at `vote_until`;
   - one standing objection stops the clock and it waits for an admin.
   The author does not vote on their own proposal.

4. Implementation job. Accepting a proposal moves the ticket to `accepted`.
   An account claims it (`ticket_claims(ticket_id, proposal_id, user_id,
   claimed_at, released_at)`), moving it to `in_progress`. One live claim per
   ticket; release or admin reassign ends it.

5. Review link. `reviews_submit` takes an optional `ticket_id`; the review row
   stores it. The ticket page lists its reviews and their standing. When a
   linked review lands, the lander writes a ticket event (not a status change);
   resolving stays a human step, as `issue_tickets.py` says ("no automatic
   issue resolution").

6. Messages. The messenger notifies a ticket's owner and voters on a new
   proposal, an objection, a decision and a landing, through the existing
   bell, as review threads already do.

## Decisions that are the admin's (not guessed here)

1. Who may vote: every signed-in account, accounts with `dev_ui`, or a
   group? (Draft assumes the same set that may remark on reviews.)
2. Vote period default and bounds. (Draft: 24 h, like reviews; max 30 days.)
3. Is silence consent for proposals too, or must a proposal gather at least
   one agree (or an admin yes) before it passes?
4. May agents (claude, codex) propose, vote and claim, or only propose?
5. Are feature tickets visible to all accounts (so they can be voted on),
   while issue tickets stay owner+admin as today?
6. Does the legacy `/api/tickets` stay separate, or fold into this?

## Not in scope

No code in this change. No change to `branch_review.standing`, to the lander,
or to permissions. Once the decisions above are made, the implementation is a
separate branch and review against this note.
