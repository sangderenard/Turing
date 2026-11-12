// Declarative control metadata extracted from marble_studio.html and the
// material-selector initialization in marble_studio.js. The drift test checks
// both sources. These are control defaults, not a full resolved lighting rig.
export const STUDIO_LIGHTING_FIELDS = {
  "key-shape": {
    "type": "select-one",
    "options": [
      "bare",
      "scrim",
      "grid",
      "snoot"
    ],
    "defaultValue": "snoot"
  },
  "light-x": {
    "type": "range",
    "min": "-5000",
    "max": "5000",
    "defaultValue": -180
  },
  "light-y": {
    "type": "range",
    "min": "-5000",
    "max": "5000",
    "defaultValue": 220
  },
  "light-z": {
    "type": "range",
    "min": "-5000",
    "max": "5000",
    "defaultValue": 220
  },
  "key-filter-left": {
    "type": "color",
    "defaultValue": "#ffffff"
  },
  "key-filter-right": {
    "type": "color",
    "defaultValue": "#ffffff"
  },
  "light-strength-left": {
    "type": "range",
    "min": "0",
    "max": "4000",
    "defaultValue": 220
  },
  "light-strength-right": {
    "type": "range",
    "min": "0",
    "max": "4000",
    "defaultValue": 0
  },
  "light-bg-strength": {
    "type": "range",
    "min": "0",
    "max": "2000",
    "defaultValue": 60
  },
  "background-filter": {
    "type": "color",
    "defaultValue": "#ffffff"
  },
  "light-kicker-left-strength": {
    "type": "range",
    "min": "0",
    "max": "2000",
    "defaultValue": 0
  },
  "kicker-left-filter": {
    "type": "color",
    "defaultValue": "#ffffff"
  },
  "light-kicker-right-strength": {
    "type": "range",
    "min": "0",
    "max": "2000",
    "defaultValue": 15
  },
  "kicker-right-filter": {
    "type": "color",
    "defaultValue": "#ffffff"
  },
  "light-fill-upper-strength": {
    "type": "range",
    "min": "0",
    "max": "2000",
    "defaultValue": 0
  },
  "fill-upper-filter": {
    "type": "color",
    "defaultValue": "#ffffff"
  },
  "light-fill-lower-strength": {
    "type": "range",
    "min": "0",
    "max": "2000",
    "defaultValue": 0
  },
  "fill-lower-filter": {
    "type": "color",
    "defaultValue": "#ffffff"
  },
  "floor-pattern": {
    "type": "select-one",
    "options": [
      "solid",
      "checker"
    ],
    "defaultValue": "solid"
  },
  "floor-material-a": {
    "type": "select-one",
    "materials": true,
    "defaultValue": "white_plaster"
  },
  "floor-material-b": {
    "type": "select-one",
    "materials": true,
    "defaultValue": "white_plaster"
  },
  "light-table-left-strength": {
    "type": "range",
    "min": "0",
    "max": "2000",
    "defaultValue": 60
  },
  "table-left-filter": {
    "type": "color",
    "defaultValue": "#ffffff"
  },
  "light-table-right-strength": {
    "type": "range",
    "min": "0",
    "max": "2000",
    "defaultValue": 60
  },
  "table-right-filter": {
    "type": "color",
    "defaultValue": "#ffffff"
  },
  "light-under-strength": {
    "type": "range",
    "min": "0",
    "max": "2000",
    "defaultValue": 15
  },
  "under-filter": {
    "type": "color",
    "defaultValue": "#ffffff"
  },
  "wall-pattern": {
    "type": "select-one",
    "options": [
      "solid",
      "checker"
    ],
    "defaultValue": "solid"
  },
  "wall-material-a": {
    "type": "select-one",
    "materials": true,
    "defaultValue": "white_plaster"
  },
  "wall-material-b": {
    "type": "select-one",
    "materials": true,
    "defaultValue": "white_plaster"
  },
  "ceiling-pattern": {
    "type": "select-one",
    "options": [
      "solid",
      "checker"
    ],
    "defaultValue": "solid"
  },
  "ceiling-material-a": {
    "type": "select-one",
    "materials": true,
    "defaultValue": "white_plaster"
  },
  "ceiling-material-b": {
    "type": "select-one",
    "materials": true,
    "defaultValue": "white_plaster"
  },
  "backdrop-layers": {
    "type": "range",
    "min": "0",
    "max": "6",
    "defaultValue": 0
  },
  "backdrop-material": {
    "type": "select-one",
    "materials": true,
    "options": [
      "calibration"
    ],
    "defaultValue": "stone_onyx"
  },
  "backdrop-sweep": {
    "type": "checkbox",
    "defaultValue": false
  },
  "checker": {
    "type": "checkbox",
    "defaultValue": true
  },
  "window-glass": {
    "type": "checkbox",
    "defaultValue": false
  },
  "key-filter-left-off": {"type":"checkbox","defaultValue":false},
  "key-filter-right-off": {"type":"checkbox","defaultValue":false},
  "background-filter-off": {"type":"checkbox","defaultValue":false},
  "kicker-left-filter-off": {"type":"checkbox","defaultValue":false},
  "kicker-right-filter-off": {"type":"checkbox","defaultValue":false},
  "fill-lower-filter-off": {"type":"checkbox","defaultValue":false},
  "fill-upper-filter-off": {"type":"checkbox","defaultValue":false},
  "table-left-filter-off": {"type":"checkbox","defaultValue":false},
  "table-right-filter-off": {"type":"checkbox","defaultValue":false},
  "under-filter-off": {"type":"checkbox","defaultValue":false},
  "room-scale": {
    "type": "range",
    "min": "1",
    "max": "8",
    "defaultValue": 6
  }
};
export function studioLightingFields(materialNames) {
  return Object.fromEntries(Object.entries(STUDIO_LIGHTING_FIELDS).map(([id,field])=>[id,
    {...field,...(field.type==='select-one'?{options:[...(field.options??[]),...(field.materials?materialNames:[])].map(value=>({value}))}:{})}]));
}
