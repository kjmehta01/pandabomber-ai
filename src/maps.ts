// Curriculum of 3 sizes × {empty, normal-wood, sparse-wood}. Empty maps boot
// players to high powerup tiers so the agent fights with real reach; sparse-
// wood maps use a mid tier to compensate for fewer harvestable powerups.
// All maps enable random spawns to keep dodge pressure on once wood is gone.

import { SimConfig } from './sim';

const WOOD_NORMAL = 0.8;
const WOOD_SPARSE = 0.27; // ~1/3 of normal

export interface MapPreset extends Omit<SimConfig, 'numPlayers' | 'seed'> {
    name: string;
}

export const MAPS: MapPreset[] = [
    // 5×7
    {
        name: '5x7-empty-pwr5',
        boardH: 5, boardW: 7,
        woodOdds: 0,
        startBombPower: 5, startMaxBombs: 5, startSpeedTier: 5,
        randomSpawnsEnabled: true,
    },
    {
        name: '5x7-wood',
        boardH: 5, boardW: 7,
        woodOdds: WOOD_NORMAL,
        randomSpawnsEnabled: true,
    },
    {
        name: '5x7-sparsewood-pwr2',
        boardH: 5, boardW: 7,
        woodOdds: WOOD_SPARSE,
        startBombPower: 2, startMaxBombs: 2, startSpeedTier: 2,
        randomSpawnsEnabled: true,
    },

    // 9×13
    {
        name: '9x13-empty-pwr7',
        boardH: 9, boardW: 13,
        woodOdds: 0,
        startBombPower: 7, startMaxBombs: 7, startSpeedTier: 7,
        randomSpawnsEnabled: true,
    },
    {
        name: '9x13-wood',
        boardH: 9, boardW: 13,
        woodOdds: WOOD_NORMAL,
        randomSpawnsEnabled: true,
    },
    {
        name: '9x13-sparsewood-pwr3',
        boardH: 9, boardW: 13,
        woodOdds: WOOD_SPARSE,
        startBombPower: 3, startMaxBombs: 3, startSpeedTier: 3,
        randomSpawnsEnabled: true,
    },

    // 13×19 (production size)
    {
        name: '13x19-wood',
        boardH: 13, boardW: 19,
        woodOdds: WOOD_NORMAL,
        randomSpawnsEnabled: true,
    },
    {
        name: '13x19-empty-pwr9',
        boardH: 13, boardW: 19,
        woodOdds: 0,
        startBombPower: 9, startMaxBombs: 9, startSpeedTier: 9,
        randomSpawnsEnabled: true,
    },
    {
        name: '13x19-sparsewood-pwr4',
        boardH: 13, boardW: 19,
        woodOdds: WOOD_SPARSE,
        startBombPower: 4, startMaxBombs: 4, startSpeedTier: 4,
        randomSpawnsEnabled: true,
    },
];

export function mapForEpisode(ep: number): MapPreset {
    return MAPS[ep % MAPS.length];
}
