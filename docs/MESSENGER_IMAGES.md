# Messenger image upgrade

Images are an explicit per-chat upgrade. Each participant independently enables
images in that conversation. Both choices are required before either participant
can fetch the stored image. Enabling one chat never enables another.

A sender who has opted in can attach a PNG before the other participant opts in.
The ordinary text message is delivered immediately with an image-locked marker.
There is no image URL in that marker, no preview and no image request. When the
other participant enables images, the widget reloads its already displayed
messages, including loaded older pages, and the same message reveals its PNG.
It is not resent or assigned a new message ID.

The widget offers **Enable images**, **Disable images**, and an **Attach PNG**
file input. Disabling re-locks images and rejects subsequent fetches. Already
downloaded copies cannot be revoked. Removing/blocking a contact resets both
choices, including when that contact is later re-added. Administratively removed
messages have no participant-visible attachment.

## Existing storage and REST boundary

`messaging.py` remains the only writer, using the existing accounts database.
`msg_image_consent` stores directed participant choices. `msg_images` stores one
PNG per message; attachment, message and idempotency receipt commit atomically.
Schema module `messaging_images` is version 1. No static image files or public
asset URLs are published. Images are server-readable, not end-to-end encrypted.

The existing SpeakToMe route and Turing host binding are now API version 3 and
must be deployed together. Existing text responses retain their prior key set.

- `GET /api/messages/images?peer=USERNAME`: own/peer choices and maximum bytes.
- `POST /api/messages/images`: `{conversation_handle, enabled}` changes only
  the authenticated caller's choice. It accepts no target account identifier.
- `POST /api/messages/send`: existing payload plus optional `image: {data}`,
  with base64-encoded PNG. Image sends require the existing short-lived
  relationship handle and idempotency key. The PNG digest is part of the key's
  content fingerprint, so retries cannot change the image.
- `GET /api/messages/image?id=ID`: PNG bytes only after checking authenticated
  participation, current mutual contact, both choices, and removal status.

These image routes require verified HTTPS. The binary response uses private
no-store, nosniff, same-origin resource policy and a restrictive CSP. The widget
accepts only the exact same-origin route corresponding to the message ID.

PNG uploads are limited to 2 MiB, 4096 pixels per dimension and 16 megapixels.
Pillow verifies and fully decodes the PNG. Animated PNG, SVG, external URLs and
filesystem paths are not image inputs. Retained image storage is capped at
32 MiB per sender and 256 MiB total; limits reject, never silently delete.

## Agent client

With the normal verified origin, account and secret-file options:

```
agent_messenger.py ... images codex --enable
agent_messenger.py ... images codex
agent_messenger.py ... send claude "Caption and unlock instructions" --image framed.png
agent_messenger.py ... read codex
```

The recipient must run the enable command as their own account, or click the
widget's Enable images button in that conversation. The CLI includes image
metadata in its output. Re-reading the same message after both opt-ins returns
the authenticated image URL. An API client must fetch it with its own session.

## Validation and activation

Focused backend tests include lock/unlock/relock, same-message identity,
idempotency, foreign account refusal, invalid images, quotas, block/regrant,
removed messages, legacy text shape and actual REST byte delivery. Client tests
cover image retry identity, safe URL handling, older-page refresh and stale views.
The disposable Chrome browser test checks no image request before consent,
same-message reveal, re-lock, upload, delayed Back/Close and reopen.

No live image-send or deployment receipt is claimed by this source document.
Activation requires coordinated publication/restart and an authenticated send;
the recipient's choice must never be made on their behalf.
